"""Tick mode on real loops: divergent and nested loops, breaks, continues, wave-wide skips, barriers inside loops and a lane mask
read as a number, each program drawn on D3D12 and compared with a CPU reference.

  (no arguments)                                    translate every program for guest waves of 32 and 64 and check the
                                                    CPU references against their own definition (Python)
  binary <llvm-mc>                                  the recorded code words match llvm-mc (gfx1013, wave32 / wavefrontsize64)
  dispatch <dxc> <offscreen_check> <adapter> [--only SUBSTR]
                                                    compile, run through --compute-store-case and compare every word
  controls <dxc> <offscreen_check> <adapter> [--only SUBSTR]
                                                    a reference with one word changed must be rejected, for every shape

Layout (the translator's compute layout): s4 = root constant 0 = the number of threads N, s5 = root constant 1 = K (a shape
parameter, here the loop bound or an offset), s6 = descriptor table (+0x10 input, +0x20 output), v0 = flat thread id. Thread t
publishes ONE word out[t] (t < 320: the runner compares 320 words; a thread beyond N leaves the sentinel). Each shape has a
reference written from the guest rule (per-lane semantics of a wave of 32 or 64, a short last wave, exec refinement on a
back edge), independent of the translator: the guest wave runs a loop while ANY of its lanes is active, a barrier waits for
every live wave of the group, and a lane that left the loop keeps its value.
"""
import x2w_engine_shim  # noqa: F401 - every gfx2hlsl import below is the opt-in engine
import argparse
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))

import startup_engine_compute_check as CC  # noqa: E402

G = CC.G
RSRC2 = CC.USER7 | CC.TGID_X
SENT = 0xC5C5C5C5
LDS_FILL = 0xCDCDCDCD
M = 0xFFFFFFFF
WORDS_PER_CASE = 320
WAVES = (32, 64)
SIZES = (1, 33, 64, 100, 256)
BIG_SIZES = SIZES + (1024,)          # the barrier shapes also run a full group of 1024 threads (32 waves of 32)
PRELUDE = 's_mov_b32 s7, 0\ns_load_dwordx8 s[8:15], s[6:7], 0x10\ns_waitcnt lgkmcnt(0)\nv_lshlrev_b32 v1, 2, v0\n'
STORE = 'buffer_store_dword v3, v1, s[12:15], 0 offen\ns_endpgm\n'


def live_lanes(t, n, wave):
    base = t // wave * wave
    return range(base, min(base + wave, n))


def tokens(wave):
    w64 = wave == 64
    return {'@VCC@': 'vcc' if w64 else 'vcc_lo', '@EXEC@': 'exec' if w64 else 'exec_lo',
            '@SAVE@': 's_and_saveexec_b64' if w64 else 's_and_saveexec_b32', '@OR@': 's_or_b64' if w64 else 's_or_b32',
            '@MASK@': 's[0:1]' if w64 else 's0', '@MASK2@': 's[2:3]' if w64 else 's2',
            '@BCNT@': 's_bcnt1_i32_b64' if w64 else 's_bcnt1_i32_b32', '@AND@': 's_and_b64' if w64 else 's_and_b32',
            '@ANDN2@': 's_andn2_b64' if w64 else 's_andn2_b32', '@MOV@': 's_mov_b64' if w64 else 's_mov_b32',
            '@WAVE@': str(wave), '@LOGW@': '6' if w64 else '5'}


def program(name, wave):
    """The assembly text of a shape for a guest wave size: prelude, body, buffer_store of v3, the optional tail."""
    text = PRELUDE + SHAPES[name][0] + STORE + TAILS.get(name, '')
    for key, value in tokens(wave).items():
        text = text.replace(key, value)
    return text


# ---- references: thread t of n, guest wave w, shape parameter k -> the word out[t] --------------------------------------
def trip(u, k):
    return (u + k) & 7


def ref_divergent_sum(t, n, w, k):
    return sum(i + t for i in range((5 * t + k) & 7))


def ref_nested(t, n, w, k):
    return sum(4 * i + j + 1 for i in range(t & 3) for j in range(((t >> 2) + k) & 3))


def ref_break(t, n, w, k):
    limit, acc, i = 20 + 3 * (t & 15) + k, 0, 0
    while True:
        i += 1
        acc += i
        if acc > limit:
            return (acc + (i << 16)) & M


def ref_continue(t, n, w, k):
    return sum(t + i for i in range(8 + k) if (t + i) & 3)


def ref_execz_inside(t, n, w, k):
    trips = 6 + k
    mine = t + 1 if (t & 15) < trips else 0
    entered = sum(1 for i in range(trips) if any((u & 15) == i for u in live_lanes(t, n, w)))
    return (mine + 1024 * entered) & M


def ref_uniform_bound(t, n, w, k):
    return (t * (k * (k - 1) // 2)) & M


def ref_readfirstlane(t, n, w, k):
    acc = 0
    for i in range((t + k) & 7):
        first = min(u for u in live_lanes(t, n, w) if ((u + k) & 7) > i)
        acc += 3 * first + i
    return acc & M


def ref_barrier_uniform(t, n, w, k):
    return sum(7 * (n - 1 - t) + i for i in range(3 + k)) & M


def ref_barrier_wave_trips(t, n, w, k):
    p = t ^ w
    trips = lambda u: 1 + ((u // w) & 3)  # noqa: E731
    acc = 0
    for i in range(trips(t)):
        acc += LDS_FILL if p >= n else 3 * p + 1 + min(i, trips(p) - 1)
    return acc & M


def ref_barrier_divergent(t, n, w, k, _cache={}):
    key = (n, w, k, LDS_FILL)
    if key not in _cache:
        tr = lambda u: (u + k) & 3  # noqa: E731
        lds, acc = {}, [0] * n
        for i in range(4):                                   # iteration i: writes, barrier, reads, barrier
            for u in range(n):
                if tr(u) > i:
                    lds[u] = 3 * u + i + 1
            for u in range(n):
                if tr(u) > i:
                    acc[u] += lds.get(u ^ w, LDS_FILL)
        _cache[key] = [a & M for a in acc]
    return _cache[key][t]


def ref_ballot_count(t, n, w, k):
    return sum(sum(1 for u in live_lanes(t, n, w) if trip(u, k) > i) for i in range(trip(t, k))) & M


def ref_ballot_exit(t, n, w, k):
    lanes = list(live_lanes(t, n, w))
    top = max(trip(u, k) for u in lanes)
    acc = 0
    for i in range(top):
        if trip(t, k) > i:
            acc += sum(1 for u in lanes if trip(u, k) > i) + i
    return acc & M


def ref_ballot_after_loop(t, n, w, k):
    inside = sum(1 for u in live_lanes(t, n, w) if (u & 3) == (k & 3))
    return (trip(t, k) * 64 + inside) & M


REFS = {
    'loop_divergent_sum': ref_divergent_sum, 'loop_nested': ref_nested, 'loop_break': ref_break,
    'loop_continue': ref_continue, 'loop_execz_inside': ref_execz_inside, 'loop_uniform_bound': ref_uniform_bound,
    'loop_readfirstlane': ref_readfirstlane, 'loop_barrier_uniform': ref_barrier_uniform,
    'loop_barrier_wave_trips': ref_barrier_wave_trips, 'loop_barrier_divergent': ref_barrier_divergent,
    't3_loop_ballot_count': ref_ballot_count, 't3_loop_ballot_exit': ref_ballot_exit,
    't3_ballot_after_loop': ref_ballot_after_loop,
}
# ---- recorded code: asm bodies per shape and llvm-mc words per (shape, guest wave) ------------------------------------------
SHAPES = {
    'loop_divergent_sum': ('v_mul_u32_u24 v5, 5, v0\nv_add_nc_u32 v5, s5, v5\nv_and_b32 v5, 7, v5\nv_mov_b32 v3, 0\nv_mov_b32 v4, 0\n@MOV@ @MASK@, @EXEC@\n.Ltop:\nv_cmp_lt_u32 @VCC@, v4, v5\n@AND@ @EXEC@, @EXEC@, @VCC@\ns_cbranch_execz .Ldone\nv_add_nc_u32 v6, v4, v0\nv_add_nc_u32 v3, v3, v6\nv_add_nc_u32 v4, 1, v4\ns_branch .Ltop\n.Ldone:\n@MOV@ @EXEC@, @MASK@\n', (0, 3)),
    'loop_nested': ('v_and_b32 v5, 3, v0\nv_lshrrev_b32 v8, 2, v0\nv_add_nc_u32 v8, s5, v8\nv_and_b32 v8, 3, v8\nv_mov_b32 v3, 0\nv_mov_b32 v4, 0\n@MOV@ @MASK@, @EXEC@\n.Louter:\nv_cmp_lt_u32 @VCC@, v4, v5\n@AND@ @EXEC@, @EXEC@, @VCC@\ns_cbranch_execz .Lodone\n@MOV@ @MASK2@, @EXEC@\nv_mov_b32 v7, 0\n.Linner:\nv_cmp_lt_u32 @VCC@, v7, v8\n@AND@ @EXEC@, @EXEC@, @VCC@\ns_cbranch_execz .Lidone\nv_lshlrev_b32 v6, 2, v4\nv_add_nc_u32 v6, v6, v7\nv_add_nc_u32 v6, 1, v6\nv_add_nc_u32 v3, v3, v6\nv_add_nc_u32 v7, 1, v7\ns_branch .Linner\n.Lidone:\n@MOV@ @EXEC@, @MASK2@\nv_add_nc_u32 v4, 1, v4\ns_branch .Louter\n.Lodone:\n@MOV@ @EXEC@, @MASK@\n', (0, 1)),
    'loop_break': ('v_and_b32 v5, 15, v0\nv_mul_u32_u24 v5, 3, v5\nv_add_nc_u32 v5, 20, v5\nv_add_nc_u32 v5, s5, v5\nv_mov_b32 v3, 0\nv_mov_b32 v4, 0\n@MOV@ @MASK@, @EXEC@\n.Ltop:\nv_add_nc_u32 v4, 1, v4\nv_add_nc_u32 v3, v3, v4\nv_cmp_gt_u32 @VCC@, v3, v5\n@ANDN2@ @EXEC@, @EXEC@, @VCC@\ns_cbranch_execnz .Ltop\n@MOV@ @EXEC@, @MASK@\nv_lshlrev_b32 v6, 16, v4\nv_add_nc_u32 v3, v3, v6\n', (0, 9)),
    'loop_continue': ('v_mov_b32 v3, 0\ns_mov_b32 s16, 0\ns_add_u32 s17, s5, 8\n.Ltop:\nv_mov_b32 v4, s16\nv_add_nc_u32 v6, v0, v4\nv_and_b32 v6, 3, v6\nv_cmp_ne_u32 @VCC@, 0, v6\n@SAVE@ @MASK@, @VCC@\ns_cbranch_execz .Lskip\nv_add_nc_u32 v6, v0, v4\nv_add_nc_u32 v3, v3, v6\n.Lskip:\n@MOV@ @EXEC@, @MASK@\ns_add_u32 s16, s16, 1\ns_cmp_lt_u32 s16, s17\ns_cbranch_scc1 .Ltop\n', (0, 1, 5)),
    'loop_execz_inside': ('v_mov_b32 v3, 0\ns_mov_b32 s16, 0\ns_mov_b32 s20, 0\ns_add_u32 s17, s5, 6\nv_and_b32 v5, 15, v0\n.Ltop:\nv_mov_b32 v4, s16\nv_cmp_eq_u32 @VCC@, v5, v4\n@SAVE@ @MASK@, @VCC@\ns_cbranch_execz .Lskip\ns_add_u32 s20, s20, 1\nv_add_nc_u32 v3, v3, v0\nv_add_nc_u32 v3, v3, 1\n.Lskip:\n@MOV@ @EXEC@, @MASK@\ns_add_u32 s16, s16, 1\ns_cmp_lt_u32 s16, s17\ns_cbranch_scc1 .Ltop\ns_lshl_b32 s20, s20, 10\nv_add_nc_u32 v3, s20, v3\n', (0, 4, 10)),
    'loop_uniform_bound': ('v_mov_b32 v3, 0\ns_mov_b32 s16, 0\n.Ltop:\ns_cmp_ge_u32 s16, s5\ns_cbranch_scc1 .Ldone\nv_mov_b32 v4, s16\nv_mul_u32_u24 v6, v4, v0\nv_add_nc_u32 v3, v3, v6\ns_add_u32 s16, s16, 1\ns_branch .Ltop\n.Ldone:\n', (0, 1, 5, 17)),
    'loop_readfirstlane': ('v_add_nc_u32 v5, s5, v0\nv_and_b32 v5, 7, v5\nv_mov_b32 v3, 0\nv_mov_b32 v4, 0\n@MOV@ @MASK@, @EXEC@\n.Ltop:\nv_cmp_lt_u32 @VCC@, v4, v5\n@AND@ @EXEC@, @EXEC@, @VCC@\ns_cbranch_execz .Ldone\nv_mul_u32_u24 v6, 3, v0\nv_add_nc_u32 v6, v4, v6\nv_readfirstlane_b32 s16, v6\nv_add_nc_u32 v3, s16, v3\nv_add_nc_u32 v4, 1, v4\ns_branch .Ltop\n.Ldone:\n@MOV@ @EXEC@, @MASK@\n', (0, 5)),
    'loop_barrier_uniform': ('v_mov_b32 v3, 0\ns_mov_b32 s16, 0\ns_add_u32 s17, s5, 3\ns_sub_u32 s18, s4, 1\nv_sub_nc_u32 v4, s18, v0\nv_lshlrev_b32 v4, 2, v4\n.Ltop:\nv_mul_u32_u24 v2, 7, v0\nv_add_nc_u32 v2, s16, v2\nds_write_b32 v1, v2\ns_waitcnt lgkmcnt(0)\ns_barrier\nds_read_b32 v6, v4\ns_waitcnt lgkmcnt(0)\nv_add_nc_u32 v3, v3, v6\ns_barrier\ns_add_u32 s16, s16, 1\ns_cmp_lt_u32 s16, s17\ns_cbranch_scc1 .Ltop\n', (0, 2)),
    'loop_barrier_wave_trips': ('v_readfirstlane_b32 s16, v0\ns_lshr_b32 s16, s16, @LOGW@\ns_and_b32 s17, s16, 3\ns_add_u32 s17, s17, 1\ns_mov_b32 s18, 0\nv_xor_b32 v4, @WAVE@, v0\nv_lshlrev_b32 v4, 2, v4\nv_mov_b32 v3, 0\n.Ltop:\nv_mul_u32_u24 v2, 3, v0\nv_add_nc_u32 v2, s18, v2\nv_add_nc_u32 v2, 1, v2\nds_write_b32 v1, v2\ns_waitcnt lgkmcnt(0)\ns_barrier\nds_read_b32 v6, v4\ns_waitcnt lgkmcnt(0)\nv_add_nc_u32 v3, v3, v6\ns_barrier\ns_add_u32 s18, s18, 1\ns_cmp_lt_u32 s18, s17\ns_cbranch_scc1 .Ltop\n', (0,)),
    'loop_barrier_divergent': ('v_add_nc_u32 v5, s5, v0\nv_and_b32 v5, 3, v5\nv_xor_b32 v7, @WAVE@, v0\nv_lshlrev_b32 v7, 2, v7\nv_mov_b32 v3, 0\nv_mov_b32 v4, 0\n@MOV@ @MASK@, @EXEC@\n.Ltop:\nv_cmp_lt_u32 @VCC@, v4, v5\n@AND@ @EXEC@, @EXEC@, @VCC@\ns_cbranch_execz .Ldone\nv_mul_u32_u24 v2, 3, v0\nv_add_nc_u32 v2, v4, v2\nv_add_nc_u32 v2, 1, v2\nds_write_b32 v1, v2\ns_waitcnt lgkmcnt(0)\ns_barrier\nds_read_b32 v6, v7\ns_waitcnt lgkmcnt(0)\nv_add_nc_u32 v3, v3, v6\ns_barrier\nv_add_nc_u32 v4, 1, v4\ns_branch .Ltop\n.Ldone:\n@MOV@ @EXEC@, @MASK@\n', (0, 1)),
    't3_loop_ballot_count': ('v_add_nc_u32 v5, s5, v0\nv_and_b32 v5, 7, v5\nv_mov_b32 v3, 0\nv_mov_b32 v4, 0\n@MOV@ @MASK@, @EXEC@\n.Ltop:\nv_cmp_lt_u32 @VCC@, v4, v5\n@AND@ @EXEC@, @EXEC@, @VCC@\ns_cbranch_execz .Ldone\n@BCNT@ s16, @EXEC@\nv_add_nc_u32 v3, s16, v3\nv_add_nc_u32 v4, 1, v4\ns_branch .Ltop\n.Ldone:\n@MOV@ @EXEC@, @MASK@\n', (0, 3)),
    't3_loop_ballot_exit': ('v_add_nc_u32 v5, s5, v0\nv_and_b32 v5, 7, v5\nv_mov_b32 v3, 0\nv_mov_b32 v4, 0\n@MOV@ @MASK@, @EXEC@\n.Ltop:\nv_cmp_lt_u32 @VCC@, v4, v5\n@BCNT@ s16, @VCC@\ns_cmp_eq_u32 s16, 0\ns_cbranch_scc1 .Ldone\n@AND@ @EXEC@, @EXEC@, @VCC@\nv_add_nc_u32 v6, s16, v4\nv_add_nc_u32 v3, v3, v6\nv_add_nc_u32 v4, 1, v4\ns_branch .Ltop\n.Ldone:\n@MOV@ @EXEC@, @MASK@\n', (0, 5)),
    't3_ballot_after_loop': ('v_add_nc_u32 v5, s5, v0\nv_and_b32 v5, 7, v5\nv_mov_b32 v3, 0\nv_mov_b32 v4, 0\n@MOV@ @MASK@, @EXEC@\n.Ltop:\nv_cmp_lt_u32 @VCC@, v4, v5\n@AND@ @EXEC@, @EXEC@, @VCC@\ns_cbranch_execz .Ldone\nv_add_nc_u32 v3, 64, v3\nv_add_nc_u32 v4, 1, v4\ns_branch .Ltop\n.Ldone:\n@MOV@ @EXEC@, @MASK@\nv_and_b32 v6, 3, v0\ns_and_b32 s17, s5, 3\nv_cmp_eq_u32 @VCC@, s17, v6\n@BCNT@ s16, @VCC@\nv_add_nc_u32 v3, s16, v3\n', (0, 2)),
}
TAILS = {}
WORDS = {
    ('loop_divergent_sum', 32): '800387be03020cf4100000fa7fc08cbf8200023485000a16050a0a4a870a0a368002067e8002087e7e0380be040b827d7e6a7e87040088bf04010c4a030d064a8108084af9ff82bf0003febe001070e001030380000081bf',
    ('loop_divergent_sum', 64): '800387be03020cf4100000fa7fc08cbf8200023485000a16050a0a4a870a0a368002067e8002087e7e0480be040b827d7e6afe87040088bf04010c4a030d064a8108084af9ff82bf0004febe001070e001030380000081bf',
    ('loop_nested', 32): '800387be03020cf4100000fa7fc08cbf8200023483000a368200102c0510104a831010368002067e8002087e7e0380be040b827d7e6a7e870e0088bf7e0382be80020e7e0711827d7e6a7e87060088bf82080c34060f0c4a810c0c4a030d064a810e0e4af7ff82bf0203febe8108084aefff82bf0003febe001070e001030380000081bf',
    ('loop_nested', 64): '800387be03020cf4100000fa7fc08cbf8200023483000a368200102c0510104a831010368002067e8002087e7e0480be040b827d7e6afe870e0088bf7e0482be80020e7e0711827d7e6afe87060088bf82080c34060f0c4a810c0c4a030d064a810e0e4af7ff82bf0204febe8108084aefff82bf0004febe001070e001030380000081bf',
    ('loop_break', 32): '800387be03020cf4100000fa7fc08cbf820002348f000a36830a0a16940a0a4a050a0a4a8002067e8002087e7e0380be8108084a0309064a030b887d7e6a7e8afbff89bf0003febe90080c34030d064a001070e001030380000081bf',
    ('loop_break', 64): '800387be03020cf4100000fa7fc08cbf820002348f000a36830a0a16940a0a4a050a0a4a8002067e8002087e7e0480be8108084a0309064a030b887d7e6afe8afbff89bf0004febe90080c34030d064a001070e001030380000081bf',
    ('loop_continue', 32): '800387be03020cf4100000fa7fc08cbf820002348002067e800390be058811801002087e00090c4a830c0c36800c8a7d6a3c80be020088bf00090c4a030d064a0003febe1081108010110abff4ff85bf001070e001030380000081bf',
    ('loop_continue', 64): '800387be03020cf4100000fa7fc08cbf820002348002067e800390be058811801002087e00090c4a830c0c36800c8a7d6a2480be020088bf00090c4a030d064a0004febe1081108010110abff4ff85bf001070e001030380000081bf',
    ('loop_execz_inside', 32): '800387be03020cf4100000fa7fc08cbf820002348002067e800390be800394be058611808f000a361002087e0509847d6a3c80be040088bf148114800301064a030025d5030301000003febe1081108010110abff4ff85bf148a148f1406064a001070e001030380000081bf',
    ('loop_execz_inside', 64): '800387be03020cf4100000fa7fc08cbf820002348002067e800390be800394be058611808f000a361002087e0509847d6a2480be040088bf148114800301064a030025d5030301000004febe1081108010110abff4ff85bf148a148f1406064a001070e001030380000081bf',
    ('loop_uniform_bound', 32): '800387be03020cf4100000fa7fc08cbf820002348002067e800390be100509bf050085bf1002087e04010c16030d064a10811080f9ff82bf001070e001030380000081bf',
    ('loop_uniform_bound', 64): '800387be03020cf4100000fa7fc08cbf820002348002067e800390be100509bf050085bf1002087e04010c16030d064a10811080f9ff82bf001070e001030380000081bf',
    ('loop_readfirstlane', 32): '800387be03020cf4100000fa7fc08cbf8200023405000a4a870a0a368002067e8002087e7e0380be040b827d7e6a7e87060088bf83000c16040d0c4a0605207e1006064a8108084af7ff82bf0003febe001070e001030380000081bf',
    ('loop_readfirstlane', 64): '800387be03020cf4100000fa7fc08cbf8200023405000a4a870a0a368002067e8002087e7e0480be040b827d7e6afe87060088bf83000c16040d0c4a0605207e1006064a8108084af7ff82bf0004febe001070e001030380000081bf',
    ('loop_barrier_uniform', 32): '800387be03020cf4100000fa7fc08cbf820002348002067e800390be05831180048192801200084c82080834870004161004044a000034d8010200007fc08cbf00008abf0000d8d8040000067fc08cbf030d064a00008abf1081108010110abff2ff85bf001070e001030380000081bf',
    ('loop_barrier_uniform', 64): '800387be03020cf4100000fa7fc08cbf820002348002067e800390be05831180048192801200084c82080834870004161004044a000034d8010200007fc08cbf00008abf0000d8d8040000067fc08cbf030d064a00008abf1081108010110abff2ff85bf001070e001030380000081bf',
    ('loop_barrier_wave_trips', 32): '800387be03020cf4100000fa7fc08cbf820002340005207e108510901083118711811180800392bea000083a820808348002067e830004161204044a8104044a000034d8010200007fc08cbf00008abf0000d8d8040000067fc08cbf030d064a00008abf1281128012110abff1ff85bf001070e001030380000081bf',
    ('loop_barrier_wave_trips', 64): '800387be03020cf4100000fa7fc08cbf820002340005207e108610901083118711811180800392bec000083a820808348002067e830004161204044a8104044a000034d8010200007fc08cbf00008abf0000d8d8040000067fc08cbf030d064a00008abf1281128012110abff1ff85bf001070e001030380000081bf',
    ('loop_barrier_divergent', 32): '800387be03020cf4100000fa7fc08cbf8200023405000a4a830a0a36a0000e3a820e0e348002067e8002087e7e0380be040b827d7e6a7e870e0088bf830004160405044a8104044a000034d8010200007fc08cbf00008abf0000d8d8070000067fc08cbf030d064a00008abf8108084aefff82bf0003febe001070e001030380000081bf',
    ('loop_barrier_divergent', 64): '800387be03020cf4100000fa7fc08cbf8200023405000a4a830a0a36c0000e3a820e0e348002067e8002087e7e0480be040b827d7e6afe870e0088bf830004160405044a8104044a000034d8010200007fc08cbf00008abf0000d8d8070000067fc08cbf030d064a00008abf8108084aefff82bf0004febe001070e001030380000081bf',
    ('t3_loop_ballot_count', 32): '800387be03020cf4100000fa7fc08cbf8200023405000a4a870a0a368002067e8002087e7e0380be040b827d7e6a7e87040088bf7e0f90be1006064a8108084af9ff82bf0003febe001070e001030380000081bf',
    ('t3_loop_ballot_count', 64): '800387be03020cf4100000fa7fc08cbf8200023405000a4a870a0a368002067e8002087e7e0480be040b827d7e6afe87040088bf7e1090be1006064a8108084af9ff82bf0004febe001070e001030380000081bf',
    ('t3_loop_ballot_exit', 32): '800387be03020cf4100000fa7fc08cbf8200023405000a4a870a0a368002067e8002087e7e0380be040b827d6a0f90be108006bf050085bf7e6a7e8710080c4a030d064a8108084af7ff82bf0003febe001070e001030380000081bf',
    ('t3_loop_ballot_exit', 64): '800387be03020cf4100000fa7fc08cbf8200023405000a4a870a0a368002067e8002087e7e0480be040b827d6a1090be108006bf050085bf7e6afe8710080c4a030d064a8108084af7ff82bf0004febe001070e001030380000081bf',
    ('t3_ballot_after_loop', 32): '800387be03020cf4100000fa7fc08cbf8200023405000a4a870a0a368002067e8002087e7e0380be040b827d7e6a7e87030088bfc006064a8108084afaff82bf0003febe83000c3605831187110c847d6a0f90be1006064a001070e001030380000081bf',
    ('t3_ballot_after_loop', 64): '800387be03020cf4100000fa7fc08cbf8200023405000a4a870a0a368002067e8002087e7e0480be040b827d7e6afe87030088bfc006064a8108084afaff82bf0004febe83000c3605831187110c847d6a1090be1006064a001070e001030380000081bf',
}

def reads_unwritten_lds(name, n, wave, k):
    """True when the guest reads an LDS word no thread has written: its value is undefined on a GPU, so the reference (which
    holds a fill word there) is checked only when changing that fill word changes no result."""
    global LDS_FILL
    saved = LDS_FILL
    try:
        first = [REFS[name](t, n, wave, k) for t in range(n)]
        LDS_FILL = 0x12345678
        return first != [REFS[name](t, n, wave, k) for t in range(n)]
    finally:
        LDS_FILL = saved


def cases(only=''):
    """(shape, guest wave, threads, K): every shape x wave x group size x K, without the cases that read undefined LDS."""
    for name, (_body, ks) in SHAPES.items():
        if only in name:
            for k in ks:
                for wave in WAVES:
                    for n in (BIG_SIZES if 'barrier' in name else SIZES):
                        if not reads_unwritten_lds(name, n, wave, k):
                            yield name, wave, n, k


def expected(name, n, wave, k):
    ref = REFS[name]
    return [(ref(t, n, wave, k) & M) if t < n else SENT for t in range(WORDS_PER_CASE)]


def case_file(name, n, wave, k, want=None):
    """offscreen_check read_store_reference layout: groups (bit 31 = initial output words follow), the 320 expected words, the
    320 initial output words (sentinel), 16 root constants (c0 = N, c1 = K), 320 source words."""
    want = expected(name, n, wave, k) if want is None else want
    return (struct.pack('<I', 1 | (1 << 31)) + struct.pack('<%dI' % WORDS_PER_CASE, *want)
            + struct.pack('<%dI' % WORDS_PER_CASE, *([SENT] * WORDS_PER_CASE))
            + struct.pack('<16I', n, k, *([0] * 14)) + struct.pack('<%dI' % WORDS_PER_CASE, *([0] * WORDS_PER_CASE)))


def translate(name, wave, n):
    code = bytes.fromhex(WORDS[(name, wave)])
    lds_units = max(1, -(-n * 4 // 512))       # GRANULATED_LDS_SIZE in 512-byte units: one word per thread
    hlsl = G.ComputePipeline(code, CC.root_blob(), RSRC2 | (lds_units << 15), (n, 1, 1), wave=wave).hlsl_cs()
    declaration = re.search(r'^RWByteAddressBuffer x2w_tickcap : register\(u\d+, space\d+\);$', hlsl, re.M)
    if declaration:     # the runner binds the store buffer only: the tick cap buffer becomes that buffer
        store = re.search(r'^RWByteAddressBuffer (\w+) : register\(u\d+\);$', hlsl, re.M).group(1)
        hlsl = hlsl.replace(declaration.group(0), '#define x2w_tickcap ' + store)
    return hlsl


def check_text():
    """The references are defined for every case and make the sentinel only for threads beyond N."""
    count = 0
    for name, wave, n, k in cases():
        want = expected(name, n, wave, k)
        assert all(v == SENT for v in want[n:]), name
        assert len(want) == WORDS_PER_CASE
        count += 1
    for name in SHAPES:
        assert name in REFS, name
        for wave in WAVES:
            assert (name, wave) in WORDS, (name, wave)
    return count


def check_translate(only=''):
    seen = set()
    for name, wave, n, _k in cases(only):
        if (name, wave, n) in seen:
            continue
        seen.add((name, wave, n))
        try:
            hlsl = translate(name, wave, n)
        except G.Unsupported as error:
            print('refused (not drawn): %s wave %d, %d threads: %s' % (name, wave, n, str(error).splitlines()[0]))
            continue
        assert 'void main' in hlsl, (name, wave, n)
    return len(seen)


def check_binary(llvm_mc):
    for (name, wave), recorded in WORDS.items():
        with tempfile.TemporaryDirectory() as tmp:
            source, obj, raw = (str(Path(tmp) / ('p' + ext)) for ext in ('.s', '.o', '.bin'))
            Path(source).write_text(program(name, wave))
            cmd = [llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-filetype=obj', '-o', obj, source]
            if wave == 64:
                cmd.insert(3, '-mattr=+wavefrontsize64')
            subprocess.run(cmd, check=True, capture_output=True)
            subprocess.run([llvm_mc.replace('llvm-mc', 'llvm-objcopy'), '-O', 'binary', '--only-section=.text', obj, raw], check=True)
            built = Path(raw).read_bytes().hex()
        if built != recorded:
            raise AssertionError('%s wave %d: assembled %s, recorded %s' % (name, wave, built, recorded))
    print('code words match llvm-mc (%d programs)' % len(WORDS))


def run(dxc, runner, adapter, directory, label, hlsl, case):
    src, cso, path = (directory / (label + ext) for ext in ('.hlsl', '.cso', '.case'))
    src.write_text(hlsl, encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    path.write_bytes(case)
    result = subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(path)], capture_output=True, text=True)
    notes = [line for line in result.stderr.splitlines() if not line.startswith('XBOX2WINDOWS_ADAPTER: skipping ')]
    return result, notes


def check_dispatch(dxc, runner, adapter, only='', controls=False):
    runner = str(Path(runner).resolve())
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    done, controlled = 0, set()
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        for name, wave, n, k in cases(only):
            try:
                hlsl = translate(name, wave, n)
            except G.Unsupported:
                continue                          # a program the translator refuses is not drawn
            label = '%s_k%d_w%d_n%d' % (name, k, wave, n)
            want = expected(name, n, wave, k)
            if controls:
                if n < 33 or k != SHAPES[name][1][-1] or (name, wave) in controlled:
                    continue                      # one rejected control per shape and wave, on a case with a short last wave
                controlled.add((name, wave))
                want = list(want)
                want[0] = (want[0] ^ 1) & M       # one word of the reference changed: the readback must differ
                result, notes = run(dxc, runner, adapter, directory, label + '_wrong', hlsl, case_file(name, n, wave, k, want))
                print(result.stdout, end='')
                if result.returncode == 77:
                    return 77
                if result.returncode != 1 or not any('expected=' in line for line in notes):
                    raise AssertionError('control %s: not rejected (exit %d) %s' % (label, result.returncode, result.stderr))
                print('control %s: rejected at %s' % (label, notes[0]))
            else:
                result, notes = run(dxc, runner, adapter, directory, label, hlsl, case_file(name, n, wave, k))
                print(result.stdout, end='')
                if result.returncode == 77:
                    return 77
                if result.returncode or notes:
                    raise AssertionError('case %s: %s' % (label, result.stderr or 'exit %d' % result.returncode))
                print('case %s: all %d words match' % (label, WORDS_PER_CASE))
            done += 1
    print('%d %s' % (done, 'controls rejected' if controls else 'cases match'))
    return 0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', nargs='*')
    parser.add_argument('--only', default='')
    args = parser.parse_args(argv)
    if not args.mode:
        print('compute tick loops: %d references, %d programs translate' % (check_text(), check_translate()))
        return 0
    mode, rest = args.mode[0], args.mode[1:]
    if mode == 'binary' and len(rest) == 1:
        check_binary(rest[0])
        return 0
    if mode in ('dispatch', 'controls') and len(rest) == 3:
        return check_dispatch(*rest, only=args.only, controls=mode == 'controls')
    parser.error('unknown arguments')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
