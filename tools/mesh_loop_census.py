"""Census of loops in gfx1013 mesh-stage programs and of where the bounded mesh lowerers first refuse them (#1998).

Reads every DXBC container under a directory, keeps the mesh-stage programs that tools/mesh_entry_aot.py decodes
(wave64, 128 threads, triangles), and prints aggregate Markdown tables only. No file name, digest, offset or
instruction text of an input reaches the output; refusal messages are printed with registers and numbers
replaced by N.

Static, per distinct program (same declared limits and decoded text):

  loop      every backward branch (target at or before the branch) is an observed back-edge; its body runs from
            the target (header) to the branch (latch). Nested: the body lies inside another back-edge's body.
            Nothing here proves that a body repeats or how often.
  barrier   the body holds an s_barrier, so the back-edge crosses a group barrier
  LDS       the body holds a ds_* instruction
  body instructions  header through latch, both included
  writes EXEC  the body writes EXEC (EXEC destination, *saveexec*, v_cmpx*). This is an upper bound on a
            loop-carried EXEC: a save/restore pair inside the body is counted too.
  latch tests EXEC  the latch is s_cbranch_execz or s_cbranch_execnz
  scalar    an SGPR is read in the body before the body writes it, and the body writes it (body order, not a
            data-flow fixed point)
  trip      constant pattern: an SCC latch whose last SCC write in the body is s_cmp of one SGPR against a
            constant, the body changes that SGPR only by s_add/s_sub of a constant, the last write before the
            header in program order is s_mov of a constant, and no other branch leaves the body or enters it
            past the header (a pattern, not a proof: dominance is not checked); unconditional latch: s_branch
            (the exit is a branch elsewhere in the body); other: any other conditional latch. "other" is not
            a finding that the trip count depends on data: it is everything the pattern does not match.

Dynamic: each lowerer runs once per program, mesh_entry_aot.lower with 128 zero geometry words and
mesh_dynamic_aot.enumerate_variants with 128 run-time words, as decoded and again with the two flat-scratch
setup writes (s_setreg_imm32_b32 to HW_REG_FLAT_SCR_LO/HI, which write nothing the lowerers model) replaced by
s_nop. The first refusal or exception is placed relative to the back-edges: at a latch (item I), in a loop
body, before the first loop, after a loop, or after execution (the lowerer's final checks). The lowerers stop at the
first refusal; later refusals are not observed. Branch offsets reach the lowerers signed, as their fixtures spell
them: LLVM 15 prints a negative SIMM16 unsigned (s_branch 65535).

  mesh_loop_census.py DIR --llvm LIBLLVM      decode with the LLVM C library, as the lowerers do
  mesh_loop_census.py DIR --llvm-mc LLVM_MC   decode with llvm-mc (the same instruction printer)
"""
import argparse
import hashlib
import re
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_dynamic_aot  # noqa: E402
import mesh_entry_aot  # noqa: E402
import pixel_aot  # noqa: E402

GEOMETRY_WORDS = 128
LOWERERS = ("entry", "dynamic")
TIERS = ("as decoded", "flat-scratch setup skipped")
FLAT_SCRATCH_SETUP = re.compile(r"s_setreg_imm32_b32 hwreg\(HW_REG_FLAT_SCR_(?:LO|HI)\), \S+")
BRANCHES = ("s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz",
            "s_cbranch_execz", "s_cbranch_execnz")
NO_SCALAR_DEST = ("s_cmp", "s_bitcmp", "s_cbranch", "s_branch", "s_waitcnt", "s_barrier", "s_sendmsg", "s_nop",
                  "s_clause", "s_inst_prefetch", "s_endpgm", "s_setprio", "s_store", "s_buffer_store", "s_delay_alu",
                  "s_sleep", "s_dcache", "s_icache", "s_ttrace", "s_setreg")
SGPR = re.compile(r"\bs(\d+)\b|\bs\[(\d+):(\d+)\]")
CONSTANT = r"-?(?:0x[0-9a-f]+|\d+)"
LOCATIONS = ("at a latch", "in a loop body", "before the first loop", "after a loop", "after execution", "lowered")


def use_llvm_mc(llvm_mc):
    """Route the lowerers' disassembly through llvm-mc (shortcut: patches two module globals for this process)."""
    def disassemble(code, library, *, cpu=b"gfx1013", offsets=False, features=b""):
        run = subprocess.run([llvm_mc, "--disassemble", "-show-encoding", "-arch=amdgcn", "-mcpu=" + cpu.decode(),
                              "-mattr=" + features.decode()], input=" ".join("0x%02x" % b for b in code),
                             capture_output=True, text=True, check=False)
        if run.returncode or "invalid instruction" in run.stderr:
            raise ValueError("llvm-mc could not decode the program")
        out, offset = [], 0
        for line in run.stdout.splitlines():
            asm, _, encoding = line.strip().partition("; encoding: [")
            if not asm or asm.startswith("."):
                continue
            out.append((offset, asm.strip()) if offsets else asm.strip())
            offset += encoding.count(",") + 1
        if offset != len(code):
            raise ValueError("llvm-mc decoded a different length")
        return out
    pixel_aot.disassemble = mesh_entry_aot.disassemble = disassemble


def signed_branches(decoded):
    """Spell each branch SIMM16 signed, as the lowerers read it."""
    out = []
    for offset, line in decoded:
        op, _, tail = line.partition(" ")
        if op in BRANCHES and tail.isdigit() and int(tail) >= 0x8000:
            line = "%s %d" % (op, int(tail) - 0x10000)
        out.append((offset, line))
    return out


def branches(decoded):
    """[(offset, target)] of every branch. Raises ValueError for a target that is not an instruction boundary."""
    starts = {offset for offset, _ in decoded}
    found = []
    for offset, line in decoded:
        op, _, tail = line.partition(" ")
        if op in BRANCHES:
            target = offset + 4 + 4 * int(tail)
            if target not in starts:
                raise ValueError("Branch target off an instruction boundary")
            found.append((offset, target))
    return found


def loops(decoded):
    """[(header, latch)] for every backward branch (an observed back-edge), in program order."""
    return [(target, offset) for offset, target in branches(decoded) if target <= offset]


def scalars(text):
    regs = set()
    for single, first, last in SGPR.findall(text):
        regs.update([int(single)] if single else range(int(first), int(last) + 1))
    return regs


def accesses(line):
    """(SGPRs read, SGPRs written, writes EXEC) of one decoded instruction."""
    op, _, tail = line.partition(" ")
    args = tail.split(", ") if tail else []
    dest = bool(args) and ((op.startswith("s_") and not op.startswith(NO_SCALAR_DEST))
                           or op in ("v_readfirstlane_b32", "v_readlane_b32")
                           or (op.startswith("v_cmp") and op.endswith("_e64")))
    written, read = (scalars(args[0]), scalars(", ".join(args[1:]))) if dest else (set(), scalars(tail))
    return read, written, "saveexec" in op or op.startswith("v_cmpx") or (dest and args[0].startswith("exec"))


def body(decoded, header, latch):
    return [(offset, line) for offset, line in decoded if header <= offset <= latch]


def trip(decoded, header, latch):
    lines = body(decoded, header, latch)
    latch_op = lines[-1][1].partition(" ")[0]
    if latch_op == "s_branch":
        return "unconditional latch"
    if latch_op not in ("s_cbranch_scc0", "s_cbranch_scc1"):
        return "other"
    scc = [line for _, line in lines[:-1] if line.startswith("s_") and not line.startswith(
        ("s_mov", "s_waitcnt", "s_nop", "s_load", "s_buffer_load", "s_barrier", "s_sendmsg", "s_branch", "s_cbranch",
         "s_clause", "s_setprio", "s_inst_prefetch", "s_cselect", "s_movk", "s_mul"))]
    match = scc and re.fullmatch(r"s_cmp_\w+ (s\d+), (%s)|s_cmp_\w+ (%s), (s\d+)" % (CONSTANT, CONSTANT), scc[-1])
    if not match:
        return "other"
    counter = match[1] or match[4]
    register = int(counter[1:])
    step = re.compile(r"s_(?:add|sub)_[iu]32 %s, %s, %s" % (counter, counter, CONSTANT))
    if any(register in accesses(line)[1] and not step.fullmatch(line) for _, line in lines[:-1]):
        return "other"
    if any((header <= offset < latch) != (header <= target <= latch)
           for offset, target in branches(decoded) if (header <= offset < latch) or (header < target <= latch)):
        return "other"  # another exit from the body, or an entry past the header
    before = [line for offset, line in decoded if offset < header and register in accesses(line)[1]]
    return ("constant pattern" if before and re.fullmatch(r"s_mov_b32 %s, %s" % (counter, CONSTANT), before[-1])
            else "other")


def shape(decoded, header, latch):
    lines = body(decoded, header, latch)
    seen, read_first, written, writes_exec = set(), set(), set(), False
    for _, line in lines:
        read, wrote, exec_write = accesses(line)
        read_first |= read - seen
        seen |= read | wrote
        written |= wrote
        writes_exec |= exec_write
    return (any(line == "s_barrier" for _, line in lines), any(line.startswith("ds_") for _, line in lines),
            writes_exec, lines[-1][1].startswith(("s_cbranch_execz", "s_cbranch_execnz")), bool(read_first & written),
            trip(decoded, header, latch))


def first_refusal(run):
    """(message, instruction offset or None) of the first refusal or exception, or (None, None) when lowered."""
    try:
        run()
    except Exception as error:  # noqa: BLE001 - a lowerer exception on an input is counted, not fatal
        offset, tb = None, error.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_name in ("steps", "exec_range") and "offset" in tb.tb_frame.f_locals:
                offset = tb.tb_frame.f_locals["offset"]
            tb = tb.tb_next
        return (str(error) if type(error) is ValueError else "exception " + type(error).__name__), offset
    return None, None


def location(message, offset, found):
    if message is None:
        return "lowered"
    if offset is None:
        return "after execution"
    if any(offset == latch for _, latch in found):
        return "at a latch"
    if any(header <= offset <= latch for header, latch in found):
        return "in a loop body"
    if not found or offset < min(header for header, _ in found):
        return "before the first loop"
    return "after a loop"


def normalise(message, line=None):
    """A refusal message with registers and numbers replaced by N; a lowerer exception names the instruction."""
    if line and message.startswith("exception "):
        message += " at " + line.partition(" ")[0]
    message = re.sub(r"\b([sv])\[\d+:\d+\]", r"\1[N:N]", re.sub(r"\b([sv])\d+\b", r"\1N", message))
    return re.sub(r"\b(?:0x[0-9a-f]+|\d+)\b", "N", message)


def census(programs):
    """Aggregate counters over an iterable of (decoded, limits) programs with signed branch offsets."""
    c = {key: Counter() for key in ("population", "loops", "shape", "latch")}
    c["length"] = {}
    for key in ("refusal", "message"):
        c[key] = {(lowerer, tier): Counter() for lowerer in LOWERERS for tier in TIERS}
    for decoded, limits in programs:
        try:
            found = loops(decoded)
        except ValueError as error:
            c["population"][str(error)] += 1
            continue
        has = "with loops" if found else "loop-free"
        c["population"][has] += 1
        if found:
            c["loops"]["back-edges"] += len(found)
            c["loops"]["programs with %s" % ("1 loop", "2 loops", "3 or more loops")[min(len(found), 3) - 1]] += 1
            for header, latch in found:
                kind = shape(decoded, header, latch)
                c["shape"][kind] += 1
                c["length"].setdefault(kind, []).append(len(body(decoded, header, latch)))
                c["latch"][dict(decoded)[latch].partition(" ")[0]] += 1
                nested = any(h <= header and latch <= l and (h, l) != (header, latch) for h, l in found)
                c["loops"]["nested back-edges" if nested else "outermost back-edges"] += 1
        lines = dict(decoded)
        skipped = [(offset, "s_nop 0" if FLAT_SCRATCH_SETUP.fullmatch(line) else line) for offset, line in decoded]
        for tier, program in zip(TIERS, (decoded, skipped)):
            runs = {"entry": lambda: mesh_entry_aot.lower(program, [0] * GEOMETRY_WORDS, limits),
                    "dynamic": lambda: mesh_dynamic_aot.enumerate_variants(program, GEOMETRY_WORDS, limits)}
            for lowerer, run in runs.items():
                message, offset = first_refusal(run)
                where = location(message, offset, found)
                c["refusal"][(lowerer, tier)][(where, has)] += 1
                if message is not None:
                    c["message"][(lowerer, tier)][(normalise(message, lines.get(offset)), has)] += 1
    return c


def read_programs(root, population, library):
    seen = set()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        blob = path.read_bytes()
        if blob[:4] != b"DXBC":
            population["files without a DXBC container"] += 1
            continue
        try:
            decoded, limits = mesh_entry_aot.decode_mesh(blob, library)
        except (ValueError, KeyError, struct.error) as error:
            population["containers not decoded: " + normalise(str(error))] += 1
            continue
        key = hashlib.sha256(repr((limits, [line for _, line in decoded])).encode()).digest()
        if key in seen:
            population["duplicate mesh programs"] += 1
            continue
        seen.add(key)
        yield signed_branches(decoded), limits


def table(title, rows, header="Programs"):
    return ["", "| %s | %s |" % (title, header), "|---|---:|"] + ["| %s | %d |" % row for row in rows]


def split_table(title, counter, keys):
    out = ["", "| %s | With loops | Loop-free |" % title, "|---|---:|---:|"]
    return out + ["| %s | %d | %d |" % (key, counter[(key, "with loops")], counter[(key, "loop-free")]) for key in keys
                  if counter[(key, "with loops")] or counter[(key, "loop-free")]]


def by_count(counter):
    keys = {key for key, _ in counter}
    return sorted(keys, key=lambda key: (-counter[(key, "with loops")] - counter[(key, "loop-free")], key))


def render(c, population):
    yes = {True: "yes", False: "no"}
    out = table("Population", sorted((population | c["population"]).items()))
    out += table("Loops", sorted(c["loops"].items()), "Count")
    out += table("Latch instruction", sorted(c["latch"].items(), key=lambda kv: (-kv[1], kv[0])), "Back-edges")
    out += ["", "| Barrier in body | LDS access in body | Body writes EXEC | Latch tests EXEC | Loop-carried scalar "
                "| Trip count | Back-edges | Body instructions (min / median / max) |",
            "|---|---|---|---|---|---|---:|---:|"]
    sizes = {key: sorted(lengths) for key, lengths in c["length"].items()}
    out += ["| %s | %s | %s | %s | %s | %s | %d | %d / %d / %d |" % (
        *(yes[flag] for flag in key[:5]), key[5], n, sizes[key][0], sizes[key][len(sizes[key]) // 2], sizes[key][-1])
            for key, n in sorted(c["shape"].items(), key=lambda kv: (-kv[1], kv[0]))]
    for tier in TIERS:
        for lowerer in LOWERERS:
            key = (lowerer, tier)
            name = "%s lowerer, %s" % (lowerer, tier)
            out += split_table("First refusal (%s)" % name, c["refusal"][key], LOCATIONS)
            out += split_table("First refusal message (%s)" % name, c["message"][key], by_count(c["message"][key]))
    return "\n".join(out[1:]) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--llvm", help="LLVM C library with the AMDGPU disassembler")
    backend.add_argument("--llvm-mc", help="llvm-mc with the AMDGPU target")
    args = parser.parse_args(argv)
    if args.llvm_mc:
        use_llvm_mc(args.llvm_mc)
    population = Counter()
    counters = census(read_programs(args.directory, population, args.llvm or args.llvm_mc))
    sys.stdout.write(render(counters, population))
    return 0


if __name__ == "__main__":
    sys.exit(main())
