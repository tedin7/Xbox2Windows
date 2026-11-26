"""Lane-exact EXEC core: the per-thread statements every translated guest wave needs, in one place.

Model (the lane-exact EXEC model, sections 1-3b and 2a): one host thread is one guest lane. A vector write is
guarded by the thread's own exec bit. Anything that depends on the other lanes of the wave comes from a gather site
(WaveSync in gfx2hlsl_engine.py); this module takes the gathered word as a variable name and never emits a gather or a
barrier itself, so none of its statements can end up under a guard some thread skips.

Range with scalar writes (rule 3b, the restore rule seen in #2603): a guest wave in which no lane is active
never runs the range, so every scalar the range writes reads as its pre-range value afterwards. Two lowerings, same answer:
  skip_flag   the range sits under `if (!skip)`; a skipped wave never assigns, so nothing needs restoring.
  run_restore every thread of the group runs the range with exec 0 for inactive lanes (a barrier or LDS access in the
              range cannot sit under a guard) and a wave with no active lane gets its pre-range values back.
DPP16 source validity (section 2a): src_valid = in_range && (FI || exec[src]); an invalid source reads 0 with BOUND_CTRL,
else the destination lane is not written; row_mask and bank_mask gate the write; the lane's own exec gates it too.
"""

EXEC_BIT = '(exec & 1u) != 0u'


def skip_decision(wave_any, name='skip'):
    """The wave decision of a forward s_cbranch_execz: `wave_any` is the gathered OR of the lanes' exec bits."""
    return ['bool %s = (%s) == 0u;' % (name, wave_any)]


def guarded_range(lines, skip='skip'):
    """The range under the wave decision (skip_flag lowering)."""
    return ['if (!%s) {' % skip] + ['    ' + line for line in lines] + ['}']


def snapshot(registers):
    """Copy the scalars a range writes before the range (run_restore lowering). registers: HLSL variable names."""
    return ['uint pre_%s = %s;' % (r, r) for r in registers]


def restore(registers, skip='skip'):
    """After a range that ran for the whole group: a wave without an active lane gets its pre-range scalars back."""
    return ['if (%s) %s = pre_%s;' % (skip, r, r) for r in registers]


def join_value(skip, pre, post):
    """Expression for a scalar at a join of the skipped path and the run path."""
    return '(%s ? %s : %s)' % (skip, pre, post)


def masked_write(dest, value, exec_expr=EXEC_BIT):
    """A vector write: only a lane whose exec bit is set writes."""
    return ['if (%s) %s = %s;' % (exec_expr, dest, value)]


def dpp_source(in_range, src_exec, fi, bound_ctrl):
    """(src_valid, read_zero, write) as Python bools for one destination lane, from the decoded instruction bits.
    in_range: the source lane index is inside the row/quad pattern; src_exec: the source lane's exec bit;
    fi: FETCH_INACTIVE; bound_ctrl: BOUND_CTRL. write is False only when the source is invalid and BOUND_CTRL is 0
    (the row_mask, bank_mask and the lane's own exec gate the write separately, see dpp_write)."""
    valid = bool(in_range) and (bool(fi) or bool(src_exec))
    if valid:
        return True, False, True
    return False, bool(bound_ctrl), bool(bound_ctrl)


def dpp_write(dst_exec, row_enabled, bank_enabled, source_write):
    """True when the destination lane is written."""
    return bool(dst_exec) and bool(row_enabled) and bool(bank_enabled) and bool(source_write)


def dpp_lane(old, src, in_range, src_exec, dst_exec, fi, bound_ctrl, row_enabled=True, bank_enabled=True):
    """The destination lane's new value for a DPP move: `old` is its VGPR, `src` the source lane's VGPR."""
    valid, zero, write = dpp_source(in_range, src_exec, fi, bound_ctrl)
    if not dpp_write(dst_exec, row_enabled, bank_enabled, write):
        return old
    return 0 if zero else src
