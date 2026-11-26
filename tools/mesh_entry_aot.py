"""Specialize a bounded, two-wave gfx1013 mesh entry for desktop D3D12.

The allocation may declare up to 128 vertices and primitives within the PSV0
output bounds; the compute fallback keeps a single three-vertex triangle.
Geometry words and dispatch ABI are static inputs. Position and transform buffers
remain GPU inputs. Only forward control flow decided by static values is accepted.
"""

import argparse
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import zlib

from mesh_clip_cull import decode_vs_out_cntl, hlsl_fields, hlsl_writes, vertex_distances
from mesh_param_link import hlsl_param_fields, hlsl_param_writes, merge_wave_params, validate_linkage
from pixel_aot import MESH_WAVE64, container_code, disassemble
from resource_inventory import parts

U32 = (1 << 32) - 1
WAVE = (1 << 64) - 1
LANE_SPILL_OPS = {"v_writelane_b32", "v_readlane_b32"}
VGPR_OPERAND = re.compile(r"\bv(?:(\d+)\b|\[(\d+):(\d+)\])")
LDS_BYTES = 1 << 16
FLAT_SCRATCH_WRITE = re.compile(r"s_setreg_imm32_b32 hwreg\(HW_REG_FLAT_SCR_(?:LO|HI)\), (.+)")
# The immediate spellings compute_store_aot accepts for the same write (a copy of its pattern).
IMM32_LITERAL = re.compile(
    r"-?(?:0x[0-9a-fA-F]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)",
    re.IGNORECASE,
)


def flat_scratch_write(line):
    """A whole-register immediate write of either half of the scratch base."""
    match = FLAT_SCRATCH_WRITE.fullmatch(line)
    return match is not None and IMM32_LITERAL.fullmatch(match[1]) is not None


def branch_target(offset, operand):
    """Byte offset a branch at `offset` jumps to. LLVM 15 prints the SIMM16 unsigned (s_branch 65535 is -1)."""
    return offset + 4 + 4 * (((int(operand) & 0xFFFF) ^ 0x8000) - 0x8000)


def span(token):
    m = re.fullmatch(r"[sv]\[(\d+):(\d+)\]", token)
    if not m or int(m[2]) < int(m[1]):
        raise ValueError("Invalid register span")
    return range(int(m[1]), int(m[2]) + 1)


# One-source binary32 forms and their HLSL lowering. round is round-half-even, like v_rndne_f32. The result of
# rcp, rsq and sqrt is whatever the host computes; callers must only rely on inputs whose result is exact.
UNARY_F32_EXPR = {
    "v_rcp_f32": "(1.0f / {x})",
    "v_rsq_f32": "rsqrt({x})",
    "v_sqrt_f32": "sqrt({x})",
    "v_floor_f32": "floor({x})",
    "v_trunc_f32": "trunc({x})",
    "v_rndne_f32": "round({x})",
    # v_fract_f32 is x - floor(x) clamped to 0x3f7fffff: a tiny negative input must not round up to 1.0. The cap is a
    # comparison, not min: D3D min returns its other operand for a NaN, and a NaN input must stay NaN.
    "v_fract_f32": "(frac({x}) >= 1.0f ? asfloat(0x3f7fffffu) : frac({x}))",
}
UNARY_F32 = {f"{name}_e{width}": expr for name, expr in UNARY_F32_EXPR.items() for width in (32, 64)}


# Inline raw descriptor constructed from the two bound entry SGPRs, a byte
# stride, identity component selects, and the observed raw-buffer flags.
VERTEX_DESCRIPTOR = [0x20000, 0x10000, U32, 0x11014FAC]
VERTEX_BYTES = 512
# A zero descriptor is the null resource; reads through it return zero.
NULL_DESCRIPTOR = [0, 0, 0, 0]
DYNAMIC_COMPARE = {"s_cmp_eq_u32": "==", "s_cmp_lt_u32": "<", "s_cmp_gt_u32": ">"}


class Dyn:
    """A scalar predicate that depends on a constant-buffer value (HLSL text)."""

    def __init__(self, expr):
        self.expr = expr


class Select:
    """A scalar register whose value depends on a Dyn predicate at a join."""

    def __init__(self, cond, when_true, when_false):
        self.cond, self.when_true, self.when_false = cond, when_true, when_false

    def __eq__(self, other):
        return (
            isinstance(other, Select)
            and self.cond is other.cond
            and self.when_true == other.when_true
            and self.when_false == other.when_false
        )


class _Undefined:
    """A scalar register no executed instruction has written on this wave.

    A range skipped by s_cbranch_execz leaves its scalar writes undone. The
    registers keep whatever the hardware held, which no export may depend on.
    """

    def __repr__(self):
        return "UNDEF"


UNDEF = _Undefined()
UNDEFINED_SCALAR = "Scalar not written on this wave (skipped range) is used"


SUPPORTED_SCALAR_OPS = frozenset(
    (
        "s_mov_b32", "s_mov_b64", "s_movk_i32", "s_cselect_b32", "s_cselect_b64",
        "s_bfe_u32", "s_lshl_b32", "s_lshr_b32", "s_add_i32", "s_add_u32",
        "s_sub_u32", "s_mul_i32", "s_and_b32", "s_and_b64", "s_or_b32",
        "s_or_b64", "s_xor_b64", "s_andn2_b64", "s_orn2_b64", "s_not_b64",
    )
)

# Lane-mask updates that keep the previous exec in the destination pair. The complemented forms negate the
# old exec, not the source: ANDN2 is SRC0 & ~EXEC and ORN2 is SRC0 | ~EXEC.
SAVEEXEC = {
    "s_and_saveexec_b64": lambda old, a: old & a,
    "s_andn2_saveexec_b64": lambda old, a: a & ~old,
    "s_or_saveexec_b64": lambda old, a: old | a,
    "s_orn2_saveexec_b64": lambda old, a: a | ~old,
    "s_xor_saveexec_b64": lambda old, a: old ^ a,
}


class Wave:
    def __init__(self, wave, geometry, vertex_header=None):
        self.wave = wave
        self.geometry = geometry
        self.s = {3: wave << 24, 17: 0x10000, 20: 0, 21: 0}
        self.vertex_header = vertex_header
        if vertex_header is not None:
            self.s.update({18: VERTEX_DESCRIPTOR[0], 19: 0})
        self.v = [{5: 0} for _ in range(64)]
        self.exec = WAVE
        self.vcc = None
        self.scc = None
        self.pending = set()
        self.pending_v = set()
        self.spill_slots = {}
        self.spill_vgprs = set()
        # First ordinary use of each spill VGPR after a spill write. Never cleared, on purpose: any later
        # reload of that VGPR is refused, which can only over-refuse, never return a stale word.
        self.spill_vector_uses = {}
        self.barriers = []
        # Group LDS: `lds` holds dwords published at an earlier barrier. A wave's own
        # stores stay in `staged` until every wave reaches the next barrier.
        self.lds = {}
        self.staged = {}
        self.reads = set()
        self.pending_ds = set()
        self.ds_writes_pending = False
        self.counts = []
        self.positions = {}
        self.attributes = {}
        self.params = {}
        self.indices = {}
        self.m0 = None
        self.written = set()
        # Position exports per vertex in program order: (target, values, done).
        self.position_exports = {}

    def scalar(self, t, allow_undef=False):
        if t.startswith("s["):
            regs = span(t)
            if len(regs) != 2:
                raise ValueError("Expected a scalar pair")
            lo, hi = (self.scalar(f"s{i}", allow_undef) for i in regs)
            if lo is UNDEF or hi is UNDEF:
                return UNDEF
            if type(lo) is not int or type(hi) is not int:
                raise ValueError("Scalar pair requires concrete words")
            return lo | hi << 32
        if re.fullmatch(r"s\d+", t):
            i = int(t[1:])
            if i in self.pending:
                raise ValueError("Scalar load used before wait")
            # A register no instruction ever writes is a bad operand, not a skipped write.
            if i not in self.written and i not in self.s:
                raise ValueError(f"Scalar register s{i} used before it is written")
            value = self.s.get(i, UNDEF)
            if value is UNDEF and not allow_undef:
                raise ValueError(UNDEFINED_SCALAR)
            return value
        if t == "exec":
            return self.exec
        if t in ("exec_lo", "exec_hi"):
            return self.exec >> (32 if t == "exec_hi" else 0) & U32
        if t == "vcc":
            if self.vcc is None:
                raise ValueError("Undefined wave predicate")
            return self.vcc
        if t == "src_scc":
            if self.scc is None:
                raise ValueError("Undefined scalar predicate in src_scc")
            if isinstance(self.scc, Dyn):
                raise ValueError("Dynamic scalar predicate in src_scc")
            return int(self.scc)
        if re.fullmatch(r"-?\d+\.\d+", t):
            return struct.unpack("<I", struct.pack("<f", float(t)))[0]
        return int(t, 0)

    def put(self, dest, value, wide=False):
        if value is UNDEF and dest in ("exec", "vcc", "m0"):
            raise ValueError(UNDEFINED_SCALAR)
        if dest == "exec":
            self.exec = value & WAVE
        elif dest in ("exec_lo", "exec_hi"):
            # A 32-bit write replaces one half of the 64-bit lane mask and keeps the other.
            if wide:
                raise ValueError("Unsupported scalar destination")
            shift = 32 if dest == "exec_hi" else 0
            self.exec = self.exec & ~(U32 << shift) | (value & U32) << shift
        elif dest == "vcc":
            self.vcc = value & WAVE
        elif dest == "m0":
            self.m0 = value & U32
        else:
            if dest.startswith("s["):
                rr = span(dest)
                if len(rr) != 2:
                    raise ValueError("Expected a pair destination")
                start = rr.start
                wide = True
            elif re.fullmatch(r"s\d+", dest):
                start = int(dest[1:])
            else:
                raise ValueError("Unsupported scalar destination")
            regs = range(start, start + (2 if wide else 1))
            if self.pending.intersection(regs):
                raise ValueError("Pending scalar destination")
            for j, i in enumerate(regs):
                self.s[i] = UNDEF if value is UNDEF else (value >> (32 * j)) & U32

    def vector_pending(self, lane, reg):
        """True while a vector load or LDS read still owes this register."""
        return (lane, reg) in self.pending_v or (lane, reg) in self.pending_ds

    def lane_value(self, t, lane):
        if re.fullmatch(r"v\d+", t):
            if self.vector_pending(lane, t):
                raise ValueError("Vector load used before wait")
            return self.v[lane][int(t[1:])]
        return self.scalar(t)

    def load(self, op, args):
        dest, base, offset = args
        regs = span(dest) if "[" in dest else [int(dest[1:])]
        rr = span(base)
        off = 0 if offset == "null" else self.scalar(offset)
        count = int(op.split("dwordx")[1]) if "dwordx" in op else 1
        if off % 4 or count != len(regs) or self.pending.intersection(regs):
            raise ValueError("Invalid scalar load")
        if op == "s_load_dwordx4":
            roles = (
                {32: 32, 96: 96, 128: 128}
                if self.vertex_header is None
                else {0: 32, 32: 96, 64: 128}
            )
            if self.scalar(base) != 0x10000 or off not in roles:
                raise ValueError("Unknown descriptor table or offset")
            data = [(roles[off], j) for j in range(4)]
        else:
            desc = [self.scalar(f"s{i}") for i in rr]
            if self.vertex_header is not None and desc == VERTEX_DESCRIPTOR:
                if off != 0 or count != 1:
                    raise ValueError(
                        "Only the specialized header word is available to scalar loads"
                    )
                self.s[regs[0]] = self.vertex_header
                self.pending.update(regs)
                return
            kind = desc[0][0] if isinstance(desc[0], tuple) else None
            first = off // 4
            if any(isinstance(d, Select) for d in desc):
                data = self.selected_words(desc, first, count)
            elif desc != [(kind, j) for j in range(4)]:
                raise ValueError("Unbound buffer")
            elif kind == 32:
                data = self.geometry[first : first + count]
            elif kind in (96, 128) and first + count <= 128:
                name = "pointData" if kind == 96 else "matrixData"
                data = [
                    f"{name}[{i // 4}].{'xyzw'[i % 4]}"
                    for i in range(first, first + count)
                ]
            else:
                raise ValueError("Unsupported uniform read")
            if len(data) != count:
                raise ValueError("Geometry read outside binding")
        self.s.update(zip(regs, data))
        self.pending.update(regs)

    def selected_words(self, desc, first, count):
        """Read constant words through a descriptor chosen by a join predicate."""
        conds = {id(d.cond): d.cond for d in desc if isinstance(d, Select)}
        if len(conds) != 1:
            raise ValueError("Descriptor words join different predicates")
        cond = next(iter(conds.values()))
        sides = [
            [getattr(d, arm) if isinstance(d, Select) else d for d in desc]
            for arm in ("when_true", "when_false")
        ]
        arms = []
        for side in sides:
            if side == NULL_DESCRIPTOR:
                arms.append(["0.0f"] * count)
                continue
            kind = side[0][0] if isinstance(side[0], tuple) else None
            if kind not in (96, 128) or side != [(kind, j) for j in range(4)]:
                raise ValueError("Unsupported descriptor in a join")
            if first + count > 128:
                raise ValueError("Unsupported uniform read")
            name = "pointData" if kind == 96 else "matrixData"
            arms.append(
                [f"{name}[{i // 4}].{'xyzw'[i % 4]}" for i in range(first, first + count)]
            )
        return [f"({cond.expr} ? {t} : {f})" for t, f in zip(*arms)]

    def join_arms(self, decoded, expressions, jumps, pc, target_offset, cond, *, branch_on_true):
        """Run both arms of a uniform branch on a constant predicate and merge them.

        Only scalar ALU and loads may appear inside an arm; the arms write no
        exports and leave EXEC, VCC and M0 as they were.
        """
        target = jumps[target_offset]
        if target <= pc:
            raise ValueError("Expected a forward mesh branch")
        end = target
        last = decoded[target - 1][1] if target - 1 >= pc else ""
        if last.startswith("s_branch "):
            join_offset = branch_target(decoded[target - 1][0], last.split()[1])
            if join_offset not in jumps or jumps[join_offset] <= target:
                raise ValueError("Expected a forward mesh branch")
            end, join = target - 1, jumps[join_offset]
        else:
            join = target
        before = (self.exec, self.vcc, self.m0)
        base = (dict(self.s), set(self.pending), set(self.pending_v))
        outcomes = []
        for start, stop in ((pc, end), (target, join)):
            self.s, self.pending = dict(base[0]), set(base[1])
            self.pending_v = set(base[2])
            self.scc = None
            for _ in self.exec_range(decoded, expressions, jumps, start, stop, arm=True):
                raise ValueError("Barrier inside a descriptor join arm")
            # Each arm is checked on its own so two changes cannot cancel.
            if (self.exec, self.vcc, self.m0) != before:
                raise ValueError("A descriptor join arm changed EXEC, VCC or M0")
            outcomes.append((self.s, self.pending, self.pending_v))
        # Outcomes are in fall-through, taken order. Select needs true, false.
        if branch_on_true:
            outcomes.reverse()
        (true_s, true_p, true_v), (false_s, false_p, false_v) = outcomes
        self.pending_v = true_v | false_v
        if true_s.keys() != false_s.keys():
            raise ValueError("A scalar register is defined in only one join arm")
        merged = {}
        for reg, value in true_s.items():
            other = false_s[reg]
            if value == other:
                merged[reg] = value
            elif isinstance(value, Select) or isinstance(other, Select):
                raise ValueError("Nested descriptor joins are not supported")
            else:
                merged[reg] = Select(cond, value, other)
        self.s, self.pending, self.scc = merged, true_p | false_p, None
        return join

    def scalar_op(self, op, args, *, dynamic_bitcmp=False):
        d, *src = args
        if op in (
            "s_load_dwordx4",
            "s_buffer_load_dword",
            "s_buffer_load_dwordx2",
            "s_buffer_load_dwordx4",
            "s_buffer_load_dwordx16",
        ):
            self.load(op, args)
            return
        if op in ("s_bitcmp0_b32", "s_bitcmp1_b32", "s_bitcmp0_b64", "s_bitcmp1_b64"):
            if len(args) != 2:
                raise ValueError(f"Expected two operands for {op}")
            wide = op.endswith("_b64")
            src0, src1 = args
            if wide and (re.fullmatch(r"s\d+", src0) or src0 in ("exec_lo", "exec_hi", "vcc_lo", "vcc_hi")):
                raise ValueError(f"Expected a 64-bit scalar operand for {op}")
            if not wide and (src0.startswith("s[") or src0 in ("exec", "vcc")):
                raise ValueError(f"Expected a 32-bit scalar operand for {op}")
            if src1.startswith("s[") or src1 in ("exec", "vcc"):
                raise ValueError(f"Expected a 32-bit bit-index operand for {op}")
            a, b = (self.scalar(token, True) for token in args)
            if a is UNDEF or b is UNDEF:
                self.scc = None
                return
            test_one = op.startswith("s_bitcmp1_")
            if dynamic_bitcmp and not wide and {type(a), type(b)} <= {int, str} and str in (type(a), type(b)):
                value = f"asuint({a})" if type(a) is str else f"{a & U32}u"
                index = f"(asuint({b}) & 31u)" if type(b) is str else f"{b & 31}u"
                cmp_op = "!=" if test_one else "=="
                self.scc = Dyn(f"(({value} & (1u << {index})) {cmp_op} 0u)")
                return
            if type(a) is not int or type(b) is not int:
                raise ValueError("Dynamic scalar predicate")
            mask = 63 if wide else 31
            width = WAVE if wide else U32
            bit = ((a & width) >> (b & mask)) & 1
            self.scc = bit == test_one
            return
        if op.startswith("s_cmp_"):
            a, b = (self.scalar(t, True) for t in args)
            if a is UNDEF or b is UNDEF:
                self.scc = None
                return
            if (
                op in DYNAMIC_COMPARE
                and {type(a), type(b)} <= {int, str}
                and str in (type(a), type(b))
            ):
                text = [f"asuint({v})" if type(v) is str else f"{v & U32}u" for v in (a, b)]
                self.scc = Dyn(f"({text[0]} {DYNAMIC_COMPARE[op]} {text[1]})")
                return
            if not isinstance(a, int) or not isinstance(b, int):
                raise ValueError("Dynamic scalar predicate")
            # 64-bit forms compare whole wave masks; 32-bit forms keep their rules.
            width = WAVE if op.endswith("_u64") else U32
            a, b = a & width, b & width
            relation = {
                "s_cmp_eq_u32": a == b,
                "s_cmp_lg_u32": a != b,
                "s_cmp_lt_u32": a < b,
                "s_cmp_gt_u32": a > b,
                "s_cmp_eq_u64": a == b,
                "s_cmp_lg_u64": a != b,
            }
            if op not in relation:
                raise ValueError(f"Unsupported scalar compare {op}")
            self.scc = relation[op]
            return
        if op in SAVEEXEC:
            a = self.scalar(src[0])
            if type(a) is not int:
                raise ValueError("Scalar ALU requires concrete operands")
            old = self.exec
            self.exec = SAVEEXEC[op](old, a) & WAVE
            self.put(d, old, True)
            self.scc = self.exec != 0
            return
        if op not in SUPPORTED_SCALAR_OPS:
            # Refuse before any undefined operand can hide an implicit EXEC/SCC/VCC effect.
            raise ValueError(f"Unsupported scalar instruction {op}")
        vals = [self.scalar(t, True) for t in src]
        if op in ("s_cselect_b32", "s_cselect_b64") and self.scc is not None:
            # The arm that is not selected is never read.
            vals = [vals[0] if self.scc else vals[-1]] * 2
        if any(v is UNDEF for v in vals):
            self.put(d, UNDEF, op.endswith("_b64"))
            if not op.startswith(("s_mov", "s_cselect", "s_mul")):
                self.scc = None
            return
        if any(type(v) is not int for v in vals):
            raise ValueError("Scalar ALU requires concrete operands")
        a = vals[0]
        b = vals[-1]
        if op in ("s_mov_b32", "s_mov_b64"):
            v = a
        elif op == "s_movk_i32":
            v = ((a & 0xFFFF) ^ 0x8000) - 0x8000
        elif op in ("s_cselect_b32", "s_cselect_b64"):
            if self.scc is None:
                raise ValueError("Undefined scalar predicate")
            if isinstance(self.scc, Dyn):
                raise ValueError("Dynamic scalar predicate")
            v = a if self.scc else b
        elif op == "s_bfe_u32":
            v = ((a & U32) >> (b & 31)) & ((1 << ((b >> 16) & 127)) - 1)
        elif op == "s_lshl_b32":
            v = a << (b & 31)
        elif op == "s_lshr_b32":
            v = (a & U32) >> (b & 31)
        elif op in ("s_add_i32", "s_add_u32"):
            v = a + b
        elif op == "s_sub_u32":
            v = a - b
        elif op == "s_mul_i32":
            v = a * b
        elif op == "s_and_b32" or op == "s_and_b64":
            v = a & b
        elif op in ("s_or_b32", "s_or_b64"):
            v = a | b
        elif op == "s_xor_b64":
            v = a ^ b
        elif op == "s_andn2_b64":
            v = a & ~b
        elif op == "s_orn2_b64":
            v = a | ~b
        elif op == "s_not_b64":
            v = ~a
        else:
            raise ValueError(f"Unsupported scalar instruction {op}")
        wide = op.endswith("_b64")
        self.put(d, v, wide)
        if op == "s_add_u32":
            self.scc = (a & U32) + (b & U32) > U32
        elif op == "s_sub_u32":
            self.scc = (a & U32) < (b & U32)
        elif not op.startswith(("s_mov", "s_cselect", "s_mul")):
            self.scc = (
                bool((~(a ^ b) & (a ^ v)) & (1 << 31))
                if op == "s_add_i32"
                else bool(v & (WAVE if wide else U32))
            )

    def lane_spill(self, op, args):
        if len(args) != 3:
            raise ValueError(f"Expected three operands for {op}")
        dest, source, lane_token = args
        if not re.fullmatch(r"-?(?:[0-9]+|0x[0-9a-fA-F]+)", lane_token):
            raise ValueError(f"{op} requires a constant integer lane: {lane_token}")
        lane = int(lane_token, 16 if "0x" in lane_token else 10)
        if not 0 <= lane < len(self.v):
            raise ValueError(f"{op} lane {lane} out of range for wave{len(self.v)}")
        writing = op == "v_writelane_b32"
        vgpr = dest if writing else source
        if not re.fullmatch(r"v\d+", vgpr):
            raise ValueError(f"{op} requires a single VGPR")
        reg = int(vgpr[1:])
        slot = (reg, lane)
        if writing:
            if any(int(token[1:]) == reg for _, token in self.pending_v):
                raise ValueError(f"{op} overlaps a pending vector load into {vgpr}")
            if source in ("exec_lo", "exec_hi"):
                value = self.exec >> (32 if source == "exec_hi" else 0)
            else:
                value = self.scalar(source)
            if not isinstance(value, (int, str)):
                raise ValueError(f"{op} requires a scalar word: {source}")
            value = value & U32 if isinstance(value, int) else value
            self.spill_vgprs.add(reg)
            self.spill_slots[slot] = value
            # Keep the lane visible if ordinary vector use follows the final reload.
            self.v[lane][reg] = value
            return
        if reg in self.spill_vector_uses:
            offset, other_op = self.spill_vector_uses[reg]
            raise ValueError(
                f"Mixed lane spill use of {vgpr}: {other_op} at {offset:#x} between write and read"
            )
        if slot not in self.spill_slots:
            raise ValueError(f"Read of unwritten lane spill slot {vgpr}[{lane}]")
        if not re.fullmatch(r"s\d+", dest):
            raise ValueError(f"{op} requires a single SGPR destination")
        scalar_reg = int(dest[1:])
        if scalar_reg in self.pending:
            raise ValueError("Pending scalar destination")
        self.s[scalar_reg] = self.spill_slots[slot]

    def image_op(self, op, tail, expressions):
        # Only the runtime lowerer's opt-in sampler heap path samples images.
        raise ValueError(f"Unsupported mesh instruction {op}")

    def vector_op(self, op, args, expressions):
        d, *src = args
        live = [i for i in range(64) if self.exec & (1 << i)]
        if op == "v_readfirstlane_b32":
            if not live:
                raise ValueError("Undefined first live lane")
            self.put(d, self.lane_value(src[0], live[0]))
            return
        if re.fullmatch(r"v_cmpx?_(?:gt|eq|le)_u32_e(?:32|64)", op):
            writes_exec = op.startswith("v_cmpx_")
            a, b = args if writes_exec else src
            mask = 0
            for lane in live:
                x, y = self.lane_value(a, lane), self.lane_value(b, lane)
                which = op.split("_")[2]
                if not isinstance(x, int) or not isinstance(y, int):
                    raise ValueError("Dynamic branch predicate")
                x, y = x & U32, y & U32
                if {"gt": x > y, "eq": x == y, "le": x <= y}[which]:
                    mask |= 1 << lane
            if writes_exec:
                self.exec = mask
            else:
                self.put(d, mask, True)
            return
        float_ops = {
            "v_mul_legacy_f32_e32",
            "v_mul_legacy_f32_e64",
            "v_fmac_f32_e32",
            "v_fmac_f32_e64",
            "v_add_f32_e32",
            "v_add_f32_e64",
            "v_sub_f32_e32",
            "v_sub_f32_e64",
            "v_subrev_f32_e32",
            "v_subrev_f32_e64",
            "v_mul_f32_e32",
            "v_mul_f32_e64",
            "v_fma_f32",
            "v_cvt_f32_u32_e32",
            "v_cvt_f32_u32_e64",
            "v_cvt_f32_i32_e32",
            "v_cvt_f32_i32_e64",
            "v_min_f32_e32",
            "v_min_f32_e64",
            "v_max_f32_e32",
            "v_max_f32_e64",
            *UNARY_F32,
        }
        if op in UNARY_F32 and len(src) != 1:
            raise ValueError(f"{op} takes one source")
        for lane in live:
            vals = [self.lane_value(t, lane) for t in src]
            if (
                op not in float_ops
                and op
                not in ("v_mov_b32_e32", "v_cndmask_b32_e64", "v_cndmask_b32_e32")
                and any(type(value) is not int for value in vals)
            ):
                raise ValueError("Vector integer ALU requires concrete operands")
            a, b = vals[:2] if len(vals) >= 2 else (vals[0], 0)
            if op == "v_mbcnt_lo_u32_b32":
                v = ((a & U32) & ((1 << min(lane, 32)) - 1)).bit_count() + b
            elif op == "v_mbcnt_hi_u32_b32":
                v = ((a & U32) & ((1 << max(lane - 32, 0)) - 1)).bit_count() + b
            elif op == "v_mov_b32_e32":
                v = a
            elif op in ("v_lshl_add_u32", "v_or3_b32"):
                v = (
                    ((a << (b & 31)) + vals[2])
                    if op == "v_lshl_add_u32"
                    else a | b | vals[2]
                )
            elif op == "v_add_nc_u32_e32":
                v = a + b
            elif op == "v_sub_nc_u32_e32":
                v = a - b
            elif op == "v_or_b32_e32":
                v = a | b
            elif op == "v_xor_b32_e32":
                v = a ^ b
            elif op == "v_min_u32_e32":
                v = min(a & U32, b & U32)
            elif op == "v_max_u32_e32":
                v = max(a & U32, b & U32)
            elif op == "v_mad_u32_u24":
                v = (a & 0xFFFFFF) * (b & 0xFFFFFF) + vals[2]
            elif op == "v_lshrrev_b32_e32":
                v = (b & U32) >> (a & 31)
            elif op == "v_lshlrev_b32_e32":
                v = b << (a & 31)
            elif op == "v_and_b32_e32":
                v = a & b
            elif op == "v_bfe_u32":
                v = ((a & U32) >> (b & 31)) & ((1 << (vals[2] & 31)) - 1)
            elif op in ("v_cndmask_b32_e64", "v_cndmask_b32_e32"):
                mask_token = src[2]
                mask = self.scalar(mask_token)
                v = b if mask & (1 << lane) else a
            elif op in float_ops:

                def fp(x):
                    return x if isinstance(x, str) else f"asfloat(0x{x & U32:08x}u)"

                x, y = fp(a), fp(b)
                if op.startswith("v_mul_legacy_f32_"):
                    expr = f"(({x} == 0 || {y} == 0) ? 0.0f : {x} * {y})"
                elif op == "v_fma_f32":
                    expr = f"mad({x}, {y}, {fp(vals[2])})"
                elif op.startswith("v_fmac_f32_"):
                    expr = f"mad({x}, {y}, {fp(self.lane_value(d, lane))})"
                elif op.startswith("v_mul_f32_"):
                    expr = f"({x} * {y})"
                elif op.startswith("v_sub_f32_"):
                    expr = f"({x} - {y})"
                elif op.startswith("v_subrev_f32_"):
                    expr = f"({y} - {x})"
                elif op.startswith("v_cvt_f32_"):
                    # The source is an integer word, never a float expression.
                    if not isinstance(a, int):
                        raise ValueError("Integer conversion requires a concrete operand")
                    word = f"0x{a & U32:08x}u"
                    expr = f"float({word})" if "_u32_" in op else f"float(asint({word}))"
                elif op.startswith(("v_min_f32_", "v_max_f32_")):
                    expr = f"{op[2:5]}({x}, {y})"
                elif op in UNARY_F32:
                    expr = UNARY_F32[op].format(x=x)
                else:
                    expr = f"({x} + {y})"
                if expr not in expressions:
                    expressions[expr] = f"f{len(expressions)}"
                v = expressions[expr]
            else:
                raise ValueError(f"Unsupported lane instruction {op}")
            if self.vector_pending(lane, d):
                raise ValueError("Pending vector destination")
            self.v[lane][int(d[1:])] = v & U32 if isinstance(v, int) else v

    def lds_op(self, op, tail):
        """One dword per live lane at a concrete, aligned, in-range LDS address.

        A load sees only dwords published at an earlier barrier. A wave's own
        unpublished store is refused rather than guessed at.
        """
        write = op == "ds_write_b32"
        match = re.fullmatch(r"(v\d+), (v\d+)(?: offset:(\d+))?", tail)
        if not match:
            raise ValueError(f"Unsupported {op} operands")
        first, second = match[1], match[2]
        offset = int(match[3] or 0)
        address_reg, data_reg, dest = (first, second, None) if write else (second, None, first)
        if offset >= LDS_BYTES:
            raise ValueError("LDS offset outside the group allocation")
        writes = {}
        for lane in range(64):
            if not self.exec & (1 << lane):
                continue
            address = self.lane_value(address_reg, lane)
            if type(address) is not int:
                raise ValueError("Dynamic LDS address")
            address += offset
            if address % 4 or not 0 <= address < LDS_BYTES:
                raise ValueError("LDS address outside the group allocation or unaligned")
            if write:
                data = self.lane_value(data_reg, lane)
                if type(data) is not int:
                    raise ValueError("LDS store of a non-integer value")
                if writes.setdefault(address, data & U32) != data & U32:
                    raise ValueError("Lanes store different values to one LDS dword")
            else:
                if address in self.staged:
                    raise ValueError("LDS load of this wave's store before a barrier")
                if address not in self.lds:
                    raise ValueError("LDS load of a dword no wave published")
                if self.vector_pending(lane, dest):
                    raise ValueError("LDS load overwrites a pending result")
                self.reads.add(address)
                self.v[lane][int(dest[1:])] = self.lds[address]
                self.pending_ds.add((lane, dest))
        if write:
            for address, data in writes.items():
                if self.staged.setdefault(address, data) != data:
                    raise ValueError("Conflicting LDS stores before a barrier")
            self.ds_writes_pending = True

    def run(self, decoded, expressions):
        run_group([self], decoded, expressions)

    def steps(self, decoded, expressions):
        """Execute the wave, yielding at each group barrier."""
        jumps = {offset: i for i, (offset, _) in enumerate(decoded)}
        for _, line in decoded:
            op, _, tail = line.partition(" ")
            if op.startswith(("s_cmp_", "s_bitcmp")):
                continue
            dest = tail.split(", ")[0]
            if re.fullmatch(r"s\d+", dest):
                self.written.add(int(dest[1:]))
            elif dest.startswith("s["):
                self.written.update(span(dest))
        if (yield from self.exec_range(decoded, expressions, jumps, 0, len(decoded))):
            return
        raise ValueError("No program exit")

    def exec_range(self, decoded, expressions, jumps, pc, stop, arm=False):
        """Execute decoded[pc:stop]; return True at s_endpgm, else False."""
        while pc < stop:
            offset, line = decoded[pc]
            pc += 1
            op, _, tail = line.partition(" ")
            if flat_scratch_write(line):
                continue
            args = tail.split(", ")
            if op in LANE_SPILL_OPS:
                self.lane_spill(op, args)
                continue
            if self.spill_vgprs:
                # Only a subsequent reload makes an ordinary use part of a spill interval.
                for single, first, last in VGPR_OPERAND.findall(tail):
                    regs = (int(single),) if single else range(int(first), int(last) + 1)
                    for reg in self.spill_vgprs.intersection(regs):
                        self.spill_vector_uses.setdefault(reg, (offset, op))
            if arm and (
                not op.startswith("s_")
                or op.startswith("s_cbranch_")
                or op in ("s_branch", "s_endpgm", "s_barrier", "s_sendmsg")
            ):
                raise ValueError("Unsupported instruction inside a descriptor join arm")
            if op == "s_endpgm":
                return True
            if op in ("s_inst_prefetch", "s_nop"):
                continue
            if op == "s_clause":
                if not re.fullmatch(r"0x[0-9a-fA-F]+|0|[1-9]\d*", tail) or not 0 <= int(tail, 0) <= 0x3F:
                    raise ValueError(f"Invalid s_clause operand: {tail}")
                continue
            if op == "s_barrier":
                if self.ds_writes_pending:
                    raise ValueError("LDS store not waited before the barrier")
                self.barriers.append(offset)
                yield
            elif op == "s_waitcnt":
                if not re.fullmatch(
                    r"(?:vmcnt\(0\)|expcnt\(0\)|lgkmcnt\(0\))(?: (?:vmcnt\(0\)|expcnt\(0\)|lgkmcnt\(0\)))*",
                    tail,
                ):
                    raise ValueError("Unsupported wait")
                if "lgkmcnt" in tail:
                    self.pending.clear()
                    self.pending_ds.clear()
                    self.ds_writes_pending = False
                if "vmcnt" in tail:
                    self.pending_v.clear()
            elif op in ("s_cbranch_scc0", "s_cbranch_scc1") and isinstance(self.scc, Dyn):
                pc = self.join_arms(
                    decoded, expressions, jumps, pc, branch_target(offset, tail),
                    self.scc, branch_on_true=op == "s_cbranch_scc1",
                )
            elif op == "s_branch" or op.startswith("s_cbranch_"):
                if (
                    op in ("s_cbranch_scc0", "s_cbranch_scc1")
                    and self.scc is None
                    or op == "s_cbranch_vccnz"
                    and self.vcc is None
                ):
                    raise ValueError("Undefined branch predicate")
                if op not in ("s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_execz", "s_cbranch_vccnz"):
                    raise ValueError(f"Unsupported mesh branch {op}")
                take = {
                    "s_branch": True,
                    "s_cbranch_scc0": not self.scc,
                    "s_cbranch_scc1": bool(self.scc),
                    "s_cbranch_execz": not self.exec,
                    "s_cbranch_vccnz": bool(self.vcc),
                }[op]
                target = jumps[branch_target(offset, tail)]
                if target < pc:
                    raise ValueError("Expected a forward mesh branch")
                if take:
                    pc = target
            elif op == "s_sendmsg" and tail == "sendmsg(MSG_GS_ALLOC_REQ)":
                if self.exec != 1 or self.wave != 0 or self.m0 is None:
                    raise ValueError("Unverified allocation")
                self.counts.append((self.m0 & 0xFFF, self.m0 >> 12))
            elif op == "buffer_load_dwordx3":
                match = re.fullmatch(
                    r"(v\[\d+:\d+\]), (v\d+), (s\[\d+:\d+\]), 0 offen", tail
                )
                if (
                    self.vertex_header is None
                    or not match
                    or len(span(match[1])) != 3
                    or [self.scalar(f"s{i}") for i in span(match[3])]
                    != VERTEX_DESCRIPTOR
                ):
                    raise ValueError("Unknown three-word vertex fetch")
                for lane in range(64):
                    if self.exec & (1 << lane):
                        address = self.lane_value(match[2], lane)
                        if (
                            type(address) != int
                            or address % 4
                            or not 4 <= address <= VERTEX_BYTES - 12
                        ):
                            raise ValueError(
                                "Three-word vertex fetch is outside the bound payload"
                            )
                        for j, reg in enumerate(span(match[1])):
                            key = (lane, f"v{reg}")
                            if self.vector_pending(*key):
                                raise ValueError(
                                    "Vertex load overwrites an outstanding result"
                                )
                            self.v[lane][reg] = (
                                f"asfloat(vertexData.Load({address + 4 * j}))"
                            )
                            self.pending_v.add(key)
            elif op == "buffer_load_dword":
                match = re.fullmatch(r"(v\d+), (v\d+), (s\[\d+:\d+\]), 0 offen", tail)
                if not match or [self.scalar(f"s{i}") for i in span(match[3])] != [
                    (32, j) for j in range(4)
                ]:
                    raise ValueError("Unknown geometry fetch")
                for lane in range(64):
                    if self.exec & (1 << lane):
                        addr = self.lane_value(match[2], lane)
                        if self.vector_pending(lane, match[1]):
                            raise ValueError("Vector load overwrites pending result")
                        if addr % 4 or not 0 <= addr // 4 < len(self.geometry):
                            raise ValueError("Geometry address out of range")
                        self.v[lane][int(match[1][1:])] = self.geometry[addr // 4]
                        self.pending_v.add((lane, match[1]))
            elif op in ("ds_write_b32", "ds_read_b32"):
                self.lds_op(op, tail)
            elif op == "exp":
                m = re.fullmatch(r"(prim|pos[0-3]|param(?:[0-9]|[12][0-9]|3[01])),? (.+?)( done)?", tail)
                if not m:
                    raise ValueError("Unsupported mesh export")
                # A position export may leave done to a later clip or cull distance export.
                if not m[3] and m[1] == "prim":
                    raise ValueError("Expected completed primitive or position export")
                operands = m[2].split(", ")
                for lane in range(64):
                    if not self.exec & (1 << lane):
                        continue
                    i = self.wave * 64 + lane
                    if m[1] == "prim" and operands[1:] == ["off"] * 3:
                        packed = self.lane_value(operands[0], lane)
                        if packed & 0xC0000000:
                            raise ValueError("Unsupported primitive flags")
                        if i in self.indices:
                            raise ValueError("Duplicate primitive")
                        self.indices[i] = [(packed >> b) & 1023 for b in (0, 10, 20)]
                    elif m[1] == "pos0" and len(operands) == 4:
                        if i in self.positions:
                            raise ValueError("Duplicate position")
                        self.positions[i] = [self.lane_value(t, lane) for t in operands]
                        self.position_exports.setdefault(i, []).append((m[1], None, bool(m[3])))
                    elif m[1].startswith("pos") and len(operands) == 4:
                        values = [None if t == "off" else self.lane_value(t, lane) for t in operands]
                        self.position_exports.setdefault(i, []).append((m[1], values, bool(m[3])))
                    elif m[1].startswith("param") and len(operands) == 4 and all(re.fullmatch(r"v\d+", t) for t in operands):
                        slot = int(m[1][5:])
                        slot_map = self.params.setdefault(slot, {})
                        if i in slot_map:
                            raise ValueError("Duplicate vertex attribute" if slot == 0 else f"Duplicate vertex parameter param{slot}")
                        vals = [self.lane_value(t, lane) for t in operands]
                        slot_map[i] = vals
                        if slot == 0:
                            self.attributes[i] = vals
                    else:
                        raise ValueError("Unsupported live export")
            elif op.startswith("image_"):
                self.image_op(op, tail, expressions)
            elif op.startswith("s_bitcmp"):
                # A runtime bit predicate is supported only as the input to a verified descriptor join.
                next_is_join_branch = not arm and pc < stop and decoded[pc][1].split(" ", 1)[0] in (
                    "s_cbranch_scc0", "s_cbranch_scc1",
                )
                self.scalar_op(op, args, dynamic_bitcmp=next_is_join_branch)
            elif op.startswith("s_"):
                self.scalar_op(op, args)
            elif op.startswith("v_"):
                self.vector_op(op, args, expressions)
            else:
                raise ValueError(f"Unsupported mesh instruction {op}")
        return False


def live_barriers_agree(first, second):
    # Wave.run only returns at S_ENDPGM. RDNA 2 ISA section 4.3 drops a retired
    # wave from later barriers. While both waves are live their PCs must match.
    return all(a == b for a, b in zip(first.barriers, second.barriers))


def check_interval(waves):
    """Validate one barrier interval of every wave; return the stores to publish."""
    published = {}
    for wave in waves:
        for address, data in wave.staged.items():
            if published.setdefault(address, data) != data:
                raise ValueError("Waves store different values to one LDS dword")
    for wave in waves:
        others = {a for other in waves if other is not wave for a in other.staged}
        if wave.reads & others:
            raise ValueError("LDS dword read and stored by different waves without a barrier")
    for wave in waves:
        wave.staged, wave.reads = {}, set()
    return published


def run_group(waves, decoded, expressions):
    """Run the waves of one group in lockstep between barriers.

    Each wave runs to its next barrier before any wave goes on. Stores become
    visible to every wave only when all of them have reached the barrier.
    """
    # Whole-register immediate writes of the scratch base lower to nothing when
    # nothing can read either half: scratch_* and flat_* memory forms and
    # s_getreg_* (which these lowerers do not lower for any register). Check every
    # instruction, including skipped arms, since the wave model does not track the
    # written value and only checks the instructions it executes.
    if any(flat_scratch_write(line) for _, line in decoded):
        for _, line in decoded:
            op = line.partition(" ")[0]
            if op.startswith(("scratch_", "flat_", "s_getreg_")):
                raise ValueError(f"Mesh scratch-base write may be observed by {op}")
    lds = {}
    for wave in waves:
        wave.lds = lds
    running = [wave.steps(decoded, expressions) for wave in waves]
    while True:
        # A wave that ends its program drops out of later barriers (RDNA 2 ISA
        # section 4.3); the waves still live release each other.
        live = []
        for steps in running:
            try:
                next(steps)
                live.append(steps)
            except StopIteration:
                pass
        if not live:
            check_interval(waves)
            return
        running = live
        published = check_interval(waves)
        lds.update(published)


def float4(values):
    components = ", ".join(v if isinstance(v, str) else f"asfloat(0x{v:08x}u)" for v in values)
    return f"float4({components})"


def emit_compute_draw(bindings, expressions, positions, indices, attributes):
    # Match the native trace layout, with optional PARAM0 data after its 80 bytes.
    # The vertex shader follows the generated triangle indices using SV_VertexID.
    lines = [
        *bindings,
        "RWByteAddressBuffer geometryOut : register(u0);",
        "ByteAddressBuffer geometryIn : register(t1);",
        "[numthreads(3,1,1)] void main(uint id:SV_GroupIndex) {",
        *(f"  precise float {name} = {expr};" for expr, name in expressions.items()),
    ]
    for i, values in positions.items():
        lines.append(f"  if (id == {i}) geometryOut.Store4({16 + i * 16}, asuint({float4(values)}));")
    for i, values in attributes.items():
        lines.append(f"  if (id == {i}) geometryOut.Store4({80 + i * 16}, asuint({float4(values)}));")
    lines.extend([
        "  if (!id) {",
        "    geometryOut.Store2(0, uint2(3,1));",
        f"    geometryOut.Store3(64, uint3({','.join(map(str, indices[0]))}));",
        "  }",
        "}",
        # The translated pixel input expects TEXCOORD0 in register zero.
        "struct Vertex {" + ("float4 attr:TEXCOORD0;" if attributes else "") + "float4 position:SV_Position;};",
        "Vertex vertex_main(uint id:SV_VertexID) {",
        "  uint index = geometryIn.Load(64 + id * 4);",
        "  Vertex v;",
        "  v.position = asfloat(geometryIn.Load4(16 + index * 16));",
    ])
    if attributes:
        lines.append("  v.attr = asfloat(geometryIn.Load4(80 + index * 16));")
    lines.extend(["  return v;", "}"])
    return "\n".join(lines) + "\n"


def lower(decoded, geometry, limits, vertex_header=None, *, compute_fallback=False, vs_out_cntl=None, param_linkage=None):
    layout = None if vs_out_cntl is None else decode_vs_out_cntl(vs_out_cntl)
    if vertex_header is not None and (
        type(vertex_header) != int or not 0 <= vertex_header <= U32
    ):
        raise ValueError("Invalid specialized vertex header")
    if (
        not isinstance(geometry, list)
        or not geometry
        or any(type(n) is not int or not 0 <= n <= U32 for n in geometry)
    ):
        raise ValueError("Geometry binding requires unsigned words")
    waves = [Wave(i, geometry, vertex_header) for i in range(2)]
    expressions = {}
    run_group(waves, decoded, expressions)
    if (
        not live_barriers_agree(*waves)
        or len(waves[0].counts) != 1
        or waves[1].counts
    ):
        raise ValueError("Inconsistent mesh synchronization or allocation")
    nv, np = waves[0].counts[0]
    pos = waves[0].positions | waves[1].positions
    params = merge_wave_params(waves, pos)
    attributes = params.get(0, {})
    idx = waves[0].indices | waves[1].indices
    # The native mesh shader writes any allocation within the PSV0 bounds. Each of the
    # 128 threads exports at most one vertex and one primitive, so every allocated
    # slot must be exported exactly once and nothing outside the allocation may be.
    if compute_fallback and (nv, np) != (3, 1):
        raise ValueError("The compute fallback translates only a single three-vertex triangle")
    if compute_fallback and (any(s != 0 for s in params) or (param_linkage is not None and param_linkage.semantic != "texcoord")):
        raise ValueError("The compute fallback translates only param0")
    if (
        not 0 < nv <= limits[0]
        or not 0 < np <= limits[1]
        or set(pos) != set(range(nv))
        or set(idx) != set(range(np))
        or any(max(tri) >= nv for tri in idx.values())
    ):
        raise ValueError("Conflicting mesh counts, vertices or primitive indices")
    distances = vertex_distances(waves[0].position_exports | waves[1].position_exports, layout)
    if compute_fallback and layout and layout.vectors:
        raise ValueError("The compute fallback has no room for clip or cull distances")
    exported_params = tuple(sorted(params))
    validate_linkage(
        exported_params,
        pixel_attrs=param_linkage.pixel_attrs if param_linkage is not None else None,
        flat=param_linkage.flat if param_linkage is not None else (),
        f16_plain=param_linkage.f16_plain if param_linkage is not None else (),
        distance_vectors=len(layout.fields()) if layout else 0,
        semantic=param_linkage.semantic if param_linkage is not None else ("param" if any(s > 0 for s in exported_params) else "texcoord"),
    )
    bindings = [
        "cbuffer Points : register(b0) { float4 pointData[32]; };",
        "cbuffer Matrices : register(b1) { float4 matrixData[32]; };",
    ]
    if vertex_header is not None:
        bindings.insert(0, "ByteAddressBuffer vertexData : register(t0);")
    if compute_fallback:
        return emit_compute_draw(bindings, expressions, pos, idx, attributes)
    lines = [
        *bindings,
        "#ifdef TRACE_EXPORTS",
        "RWByteAddressBuffer trace : register(u0);",
        "#endif",
        "struct Vertex {float4 position:SV_Position;" + hlsl_param_fields(exported_params, param_linkage)
        + hlsl_fields(layout) + "};",
        '[outputtopology("triangle")]',
        f"[numthreads(128,1,1)] void main(uint id:SV_GroupIndex, out vertices Vertex v[{nv}], out indices uint3 p[{np}]) {{",
        f"  SetMeshOutputCounts({nv},{np});",
    ]
    for expr, name in expressions.items():
        lines.append(f"  precise float {name} = {expr};")
    for i, values in pos.items():
        lines.extend(
            [
                f"  if(id=={i}) {{ float4 q={float4(values)}; v[{i}].position=q;",
                "#ifdef TRACE_EXPORTS",
                f"    trace.Store4({16 + i * 16},asuint(q));",
                "#endif",
                "  }",
            ]
        )
        param_writes = hlsl_param_writes(f"v[{i}]", {s: d[i] for s, d in params.items()}, exported_params, float4, param_linkage)
        if param_writes:
            if len(param_writes) == 1 and param_writes[0].startswith(f"v[{i}].attr = "):
                lines.append(f"  if(id=={i}) {param_writes[0].replace(' = ', '=')}")
            else:
                lines.append(f"  if(id=={i}) {{ {' '.join(param_writes)} }}")
        if i in distances:
            lines.append(f"  if(id=={i}) {{ {hlsl_writes(f'v[{i}]', distances[i])} }}")
    for i, tri in idx.items():
        lines.extend(
            [
                f"  if(id=={i}) {{ uint3 t=uint3({','.join(map(str, tri))}); p[{i}]=t;",
                "#ifdef TRACE_EXPORTS",
                f"    trace.Store3({16 + nv * 16 + i * 12},t);",
                "#endif",
                "  }",
            ]
        )
    lines.extend(
        [
            "#ifdef TRACE_EXPORTS",
            f"  if (!id) trace.Store2(0,uint2({nv},{np}));",
            "#endif",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


STAGE_NAMES = {
    0: "pixel",
    1: "vertex",
    2: "geometry",
    3: "hull",
    4: "domain",
    5: "compute",
    6: "library",
    13: "mesh",
    14: "amplification",
}


def require_mesh_stage(chunks):
    """Reject any other stage by name before the mesh profile checks."""
    psv = chunks[b"PSV0"]
    program = chunks[b"DXIL"]
    kind = struct.unpack_from("<H", program, 2)[0] if len(program) >= 4 else None
    for stage in (psv[28] if len(psv) > 28 else None, kind):
        if stage == 13:
            continue
        if stage == 14:
            raise ValueError(
                "Unsupported amplification stage: only the bounded mesh stage is translated"
            )
        if stage is not None:
            name = STAGE_NAMES.get(stage, f"stage {stage}")
            raise ValueError(
                f"Unsupported {name} stage: only the bounded mesh stage is translated"
            )


def decode_mesh(blob, library):
    chunks = parts(blob)
    require_mesh_stage(chunks)
    p = chunks[b"PSV0"]
    if (
        len(p) < 52
        or not 48 <= struct.unpack_from("<I", p)[0] <= len(p) - 4
        or p[28] != 13
        or p[31] != 2
        or struct.unpack_from("<3I", p, 40) != (128, 1, 1)
    ):
        raise ValueError("Expected triangle mesh, 128 threads")
    limits = struct.unpack_from("<HH", p, 16)
    if any(not 0 < n <= 256 for n in limits):
        raise ValueError("Invalid mesh declaration")
    code = container_code(chunks[b"DXIL"], stage=13, llvm=library, features=MESH_WAVE64)
    return disassemble(code, library, offsets=True, features=MESH_WAVE64), limits


def translate(blob, geometry, library, vertex_header=None, *, compute_fallback=False, vs_out_cntl=None, param_linkage=None):
    decoded, limits = decode_mesh(blob, library)
    return lower(decoded, geometry, limits, vertex_header, compute_fallback=compute_fallback,
                 vs_out_cntl=vs_out_cntl, param_linkage=param_linkage)


def _same_path(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and left.samefile(right)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("container", type=Path)
    p.add_argument(
        "--vertex-header",
        type=lambda s: int(s, 0),
        help="Specialize the indexed geometry ABI using this first word of the 512-byte vertex SRV",
    )
    p.add_argument("--geometry-words", required=True, type=Path)
    p.add_argument("--llvm", required=True)
    p.add_argument(
        "--compute-fallback",
        action="store_true",
        help="Emit SM 6.0 compute (main) and vertex (vertex_main) entry points",
    )
    p.add_argument(
        "--pa-cl-vs-out-cntl",
        type=lambda s: int(s, 0),
        help="PA_CL_VS_OUT_CNTL word that places clip and cull distances in pos1/pos2",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.output.exists() and a.output.is_dir():
            raise ValueError(f"Output path is an existing directory: {a.output}")
        if a.output.suffix.lower() != ".hlsl":
            raise ValueError("Expected a separate HLSL output")
        if _same_path(a.output, a.container) or _same_path(a.output, a.geometry_words):
            raise ValueError("Expected a separate HLSL output")
        src = translate(
            a.container.read_bytes(),
            json.loads(a.geometry_words.read_text(encoding="utf-8")),
            a.llvm,
            a.vertex_header,
            compute_fallback=a.compute_fallback,
            vs_out_cntl=a.pa_cl_vs_out_cntl,
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=a.output.parent) as staging_dir:
            staged = Path(staging_dir) / a.output.name
            staged.write_text(src, encoding="ascii")
            os.replace(staged, a.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, struct.error, zlib.error) as exc:
        p.exit(1, f"{type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
