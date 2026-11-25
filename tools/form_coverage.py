"""Count which instruction forms the translation tests cover.

Usage:
    ctest --preset dev --no-tests=error --output-junit <repo>/out/test-results.xml
    python tools/form_coverage.py out/test-results.xml [--records DIR] [--json out/coverage.json]

This measures instruction-form coverage only: which forms from a reference list appear in a
source-built fixture whose translation test passed. It is not a measure of how complete a port is;
covering a form in one fixture does not prove every operand variant or any title behavior.

Rules:
- A fixture's forms are covered only when its test passed. A skipped or failed test (or one that is
  not registered) leaves them uncovered; forms held back only by a skip are listed separately.
- x64: new fixtures declare their test in a first-line MASM comment (see CONTRIBUTING.md).
  Existing CPU_FIXTURES classifications take precedence. Unclassified files are reported.
  Fixtures registered in record mode instead require decoded-instruction records
  emitted by their differential test during the JUnit run; source macro tokens are not counted.
- gfx: only programs the translator accepted count. tests/pixel_aot_check.py records the first word
  of each line of every program tools/pixel_aot.py lowered without an error, in the file CTest names
  in FORM_COVERAGE_RECORD (<build>/form_coverage/<test>.txt). Programs a check expects to be rejected
  raise before they are recorded. A record not written during the JUnit file's run is ignored, so a
  record and a test status always come from the same run. A --records path that is not a directory is an
  error (exit 2); a directory whose records all come from another run is measured with a warning.
- Reference lists (denominators) live in docs/coverage/ and are produced by tools/form_reference.py.
Python standard library only.
"""
import argparse
from datetime import datetime
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "coverage"
RECORDS = ROOT / "out" / "build" / "dev" / "form_coverage"  # CMakePresets.json "dev" binaryDir

# The test that proves each CPU fixture was translated. Fixtures that run unchanged (not translated)
# map to None: their forms are known but never covered. An unlisted .asm is reported as unclassified.
CPU_FIXTURES = {
    "branch": "cpu_leaf_aot_branches",
    "rcx_branches": "cpu_aot_rcx_branches",
    "direct": "cpu_aot_direct_call",
    "chained": "cpu_aot_chained_ranges",
    "callback": "cpu_aot_object_callback",
    "x87_add": "cpu_aot_x87_add",
    "x87_mul": "cpu_aot_x87_mul",
    "x87_div": "cpu_aot_x87_div",
    "x87_sub": "cpu_aot_x87_sub",
    "x87_moves": "cpu_aot_x87_moves",
    "x87_harness": None,  # native x87 entry and exit state for x87_add
    "destroy": "cpu_aot_destroy_contract",
    "create_context": "cpu_aot_context_lifecycle",
    "bink_heap": "cpu_aot_bink_heap",
    "heap_layout": "cpu_aot_heap_layout",
    "shift_count": "cpu_aot_shift_count",
    "carry_rotates": "cpu_aot_carry_rotates",
    "version_query": "cpu_aot_version_query",
    "bink_descriptors": "cpu_bink_descriptors",
    "switch": "cpu_aot_jump_tables",
    "flag_conditions": "cpu_aot_flag_conditions",
    "dead_read": "cpu_aot_dead_reads",
    "switch_addr32": "cpu_aot_switch_addr32",
    "fourth_arg": "cpu_aot_fourth_argument",
    "repeat_call": "cpu_aot_repeated_call",
    "working_set_delta": "cpu_working_set_delta",
    "xlatb": "cpu_aot_xlatb",
    "rdtsc": "cpu_aot_rdtsc",
    "rdtscp": "cpu_aot_rdtscp",
    "rdtsc_harness": None,  # native CPUID and serialized reads
    "rdrand": "cpu_aot_rdrand",
    "rdseed": "cpu_aot_rdseed",
    "rdrand_harness": None,  # native CPUID and RFLAGS reads
    "cpuid": "cpu_aot_cpuid",
    "xgetbv": "cpu_aot_xgetbv",
    "cpuid_harness": None,  # native CPUID, OSXSAVE and XGETBV reads
    "mxcsr_harness": None,  # native MXCSR reads and writes
    "sse_fp_harness": None,  # native MXCSR driver for cpu_aot_sse_fp
    "exe_entry": "cpu_aot_exe_entry",
    "xchg": "cpu_aot_xchg",
    "address_size": "cpu_aot_address_size",
    "avx2_regs": "cpu_aot_avx2_regs",
    "avx_int_regs": "cpu_aot_avx_int_regs",
    "avx_int_ymm_addsub": "cpu_aot_avx_int_ymm_addsub",
    "avx_int_ymm_minmax": "cpu_aot_avx_int_ymm_minmax",
    "avx_int_ymm_sat": "cpu_aot_avx_int_ymm_sat",
    "avx_move_regs": "cpu_aot_avx_move_regs",
    "avx_move_mem": "cpu_aot_avx_move_mem",
    "call_pair": "cpu_aot_call_pair",
    "movbe": "cpu_aot_movbe",
    "bmi1": "cpu_aot_bmi1",
    "rotates_shifts": "cpu_aot_rotates_shifts",
    "frame": "cpu_aot_stack_frames",
    "calls": "cpu_aot_direct_calls",
    "call_entries": "cpu_aot_call_entries",
    "bare_calls": "cpu_aot_bound_callees",
    "string_ops": "cpu_aot_string_ops",
    "wide_arith": "cpu_aot_wide_arithmetic",
    "dead_div": "cpu_aot_dead_divides",
    "register_probe": None,  # measurement harness
    "two_imports": None,     # mapped and run unchanged by pe_map
    "exe_start": None,       # mapped and run unchanged by pe_map
    "reloc_route": None,     # mapped and run unchanged by pe_map
    "seh_route": None,       # mapped and run unchanged by pe_map
    "tls_route": None,       # mapped and run unchanged by pe_map
}

# These sources use parameterized macros. Count decoded instructions recorded by
# their passing differential test, rather than guessing the macro expansion.
CPU_RECORD_FIXTURES = {"flag_conditions"}

MASM_DIRECTIVES = {"ifdef", "ifndef", "if", "else", "elseif", "elseifdef", "endif",
                   "ifidni", "rept", "end", "public", "extern", "externdef", "include", "includelib",
                   "option", "align", "even", "org", "assume", "local"}
MASM_DEFINERS = {"proc", "endp", "label", "equ", "db", "dw", "dd", "dq", "byte", "word", "dword",
                 "qword", "segment", "ends", "struct", "macro", "endm", "textequ"}
MASM_PREFIXES = {"lock", "rep", "repe", "repz", "repne", "repnz"}
MASM_LABEL = re.compile(r"^\s*(?:@@|[@$?A-Za-z_][\w@$?]*)::?")
# MASM condition-code spellings -> iced-x86's canonical mnemonic suffix.
CONDITION_ALIASES = {"z": "e", "nz": "ne", "c": "b", "nae": "b", "nc": "ae", "nb": "ae", "nbe": "a",
                     "na": "be", "nge": "l", "nl": "ge", "nle": "g", "ng": "le", "pe": "p", "po": "np"}


def masm_mnemonics(text):
    """Instruction mnemonics in MASM source; both arms of IFDEF/ELSE count (both variants are built)."""
    found = set()
    ignored = MASM_DIRECTIVES | MASM_DEFINERS
    for line in text.splitlines():
        # Local (`@@:`) or global label, then a MASM assembly-time assignment.
        line = line.partition(";")[0]
        if ":" in line:
            line = MASM_LABEL.sub("", line)
        words = line.lower().split()
        if len(words) > 1 and words[1] == "=":
            continue
        if len(words) > 1 and words[1] == "macro":
            ignored.add(words[0])
            continue
        if (not words or words[0].startswith(".") or words[0] in ignored
                or (len(words) > 1 and words[1] in MASM_DEFINERS and words[2:3] != ["ptr"])):
            continue
        while words and words[0] in MASM_PREFIXES:
            words = words[1:]
        if not words:
            continue
        if words[0] == "int" and len(words) > 1 and masm_integer(words[1]) == 3:
            # ML64 has no INT3 mnemonic and encodes INT 3 (also written 3h or 03h) as the one-byte CC form,
            # the reference's int3.
            found.add("int3")
            continue
        if words[0] == "fsetpm":
            # ML64 has no FNSETPM spelling and assembles FSETPM as DB E4 without a WAIT (9B), the
            # reference's fnsetpm. Its other WAIT spellings (FCLEX, FINIT, FSTSW, ...) do emit the 9B.
            found.add("fnsetpm")
            continue
        found.add(x64_canonical(words[0]))
    return found


def masm_integer(token):
    """A MASM integer literal (decimal, or hexadecimal with an h suffix), or None."""
    try:
        return int(token[:-1], 16) if token.endswith("h") else int(token, 10)
    except ValueError:
        return None


def x64_canonical(mnemonic):
    for prefix in ("cmov", "set", "j"):
        if mnemonic.startswith(prefix) and mnemonic[len(prefix):] in CONDITION_ALIASES:
            return prefix + CONDITION_ALIASES[mnemonic[len(prefix):]]
    return mnemonic


def gfx_forms(mnemonics, reference):
    """Reference forms a fixture mnemonic selects: _e32 is the plain encoding, _e64 is VOP3."""
    wanted = {m if m.endswith("_e64") else m.removesuffix("_e32") for m in mnemonics}
    return {form for form in reference if form[1] in wanted}


def x64_forms(mnemonics, reference):
    return {form for form in reference if form[1] in mnemonics}


def load_reference(name):
    lines = (REFERENCE / name).read_text(encoding="utf-8").splitlines()
    return {tuple(line.split("\t")) for line in lines if line and not line.startswith("#")}


def read_results(junit):
    """CTest JUnit status per test (pass, fail or skip; CTest writes skips as status="notrun"), and the
    run's (start, end) in seconds since the epoch. CTest writes the start as local time in whole seconds."""
    suite = ET.parse(junit).getroot()
    if suite.tag != "testsuite":
        # Records are matched to the run's time window, so a file merging several runs cannot be used.
        raise ValueError(f"expected one CTest <testsuite> root, found <{suite.tag}>")
    try:
        start = datetime.fromisoformat(suite.get("timestamp")).timestamp()
    except (TypeError, ValueError):
        raise ValueError(f"<testsuite> timestamp {suite.get('timestamp')!r} is not an ISO date and time") from None
    try:
        duration = float(suite.get("time"))
    except (TypeError, ValueError):
        raise ValueError(f"<testsuite> time {suite.get('time')!r} is not a number of seconds") from None
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f"<testsuite> time {suite.get('time')!r} must be finite and nonnegative")
    status = {"run": "pass", "fail": "fail", "notrun": "skip"}
    results = {}
    for case in suite.iter("testcase"):
        name = case.get("name")
        if not name or not name.strip():
            raise ValueError("CTest <testcase> needs a nonempty name")
        if name in results:
            # Coverage records have only the test name, not its CTest directory.
            raise ValueError(f"duplicate CTest <testcase> name {name!r}; coverage records require unique names")
        state = status.get(case.get("status"), "fail")
        if case.find("failure") is not None or case.find("error") is not None:
            state = "fail"
        elif state == "pass" and case.find("skipped") is not None:
            state = "skip"
        results[name] = state
    return results, (start - 1, start + duration + 2)


def read_records(directory, window):
    """{test: first words of the accepted program lines} from records written within window."""
    start, end = window
    return {path.stem: set(path.read_text(encoding="utf-8").split())
            for path in sorted(Path(directory).glob("*.txt")) if start <= path.stat().st_mtime <= end}


def fixture_status(results, tests):
    """pass if any listed test passed, skip if none passed but one skipped, otherwise fail/absent."""
    states = [results.get(t) for t in tests if t]
    if "pass" in states:
        return "pass"
    return "skip" if "skip" in states else "fail" if "fail" in states else "absent"


def summarize(reference, fixtures, select, *, include_gaps=False):
    """fixtures: {name: (mnemonics, status)}. Returns covered/known counts and per-group detail."""
    covered = set().union(set(), *(select(m, reference) for m, s in fixtures.values() if s == "pass"))
    skipped = set().union(set(), *(select(m, reference) for m, s in fixtures.values() if s == "skip")) - covered
    known = set().union(set(), *(select(m, reference) for m, _ in fixtures.values()))
    groups = {}
    for group, form in sorted(reference):
        g = groups.setdefault(group, {"covered": 0, "total": 0})
        g["total"] += 1
        g["covered"] += (group, form) in covered
    unmatched = sorted(m for m in set().union(set(), *(m for m, _ in fixtures.values()))
                       if not select({m}, reference))
    result = {"covered": len(covered), "total": len(reference), "known_in_fixtures": len(known),
            "blocked_by_skip": sorted(f"{g} {f}" for g, f in skipped), "groups": groups,
            "fixture_forms_not_in_reference": unmatched}
    if include_gaps:
        gaps = []
        for form in sorted(reference - covered):
            evidence = "unobserved"
            if form in skipped:
                evidence = "skipped"
            elif form in known:
                evidence = "observed_without_pass"
            gaps.append((*form, evidence))
        result["gaps"] = gaps
    return result


def measure(results, root=ROOT, records=None, *, gaps_for=None):
    """results: {test: status}. root: the repository whose tests/cpu is scanned (the check passes a
    temporary one). records: {test: accepted gfx mnemonics} from read_records."""
    cpu = {}
    records = records or {}
    cpu_tests = set(CPU_FIXTURES.values()) - {None}
    for asm in sorted((root / "tests" / "cpu").glob("*.asm")):
        source = asm.read_text(encoding="utf-8-sig")
        record = asm.stem in CPU_RECORD_FIXTURES
        if asm.stem in CPU_FIXTURES:
            test = CPU_FIXTURES[asm.stem]
            status = fixture_status(results, [test]) if test else "not translated"
        else:
            header = re.fullmatch(r"; coverage: (source|record) (cpu_[a-zA-Z0-9_]+)", source.partition("\n")[0].strip())
            test = header[2] if header else None
            record = bool(header and header[1] == "record")
            status = fixture_status(results, [test]) if test else "unclassified"
            if test:
                cpu_tests.add(test)
        if record:
            forms = {x64_canonical(m) for m in records.get(test, set())}
        else:
            forms = masm_mnemonics(source)
        cpu[asm.name] = (forms, status)
    shaders = {test: (forms, fixture_status(results, [test]))
               for test, forms in records.items() if test not in cpu_tests}
    x64 = load_reference("x64_zen2.tsv")
    fixtures = {name: status for name, (_, status) in {**cpu, **shaders}.items()}
    return {
        "note": "Instruction-form coverage of source-built fixtures; not a measure of port completion.",
        "tests": {s: sum(v == s for v in results.values()) for s in ("pass", "fail", "skip")},
        "fixtures": fixtures,
        "unclassified": sorted(name for name, status in fixtures.items() if status == "unclassified"),
        "shader_records": sorted(shaders),
        "x64_zen2": summarize(x64, cpu, x64_forms, include_gaps=gaps_for == "x64_zen2"),
        "gfx1013": summarize(load_reference("gfx1013.tsv"), shaders, gfx_forms, include_gaps=gaps_for == "gfx1013"),
        "gfx1030": summarize(load_reference("rdna2.tsv"), shaders, gfx_forms, include_gaps=gaps_for == "gfx1030"),
    }


def report(m):
    t = m["tests"]
    lines = [m["note"], f"Tests: {t['pass']} passed, {t['fail']} failed, {t['skip']} skipped.", ""]
    for key, label in (("x64_zen2", "x64 (Zen 2 user-mode mnemonics)"), ("gfx1013", "gfx1013 forms"),
                       ("gfx1030", "gfx1030 (RDNA2) forms")):
        r = m[key]
        lines.append(f"{label}: {r['covered']} of {r['total']} covered "
                     f"({r['known_in_fixtures']} appear in fixtures; {len(r['blocked_by_skip'])} held back by skips)")
    lines += ["", "Shader translation records from this run: "
              + (", ".join(m["shader_records"]) or "none, so no gfx form is covered")]
    if m["unclassified"]:
        lines.append("Not counted until classified: " + ", ".join(m["unclassified"]))
    return "\n".join(lines)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("junit", type=Path)
    parser.add_argument("--records", type=Path, default=RECORDS,
                        help="directory of shader translation records (default: %(default)s)")
    parser.add_argument("--json", type=Path, help="also write the full result, per group, as JSON")
    parser.add_argument("--gaps", choices=("x64_zen2", "gfx1013", "gfx1030"),
                        help="print uncovered forms for one architecture as tab-separated group, form and evidence")
    args = parser.parse_args(argv)
    try:
        results, window = read_results(args.junit)
    except (OSError, ET.ParseError, ValueError) as error:
        print(f"form_coverage: {args.junit}: {error}", file=sys.stderr)
        return 2
    # Reading nothing from a mistyped or wrong --records path would silently count every record-mode and shader
    # form as uncovered, so a path that is not a directory is an error.
    if not args.records.is_dir():
        problem = "records path is not a directory" if args.records.exists() else "records directory does not exist"
        print(f"form_coverage: {args.records}: {problem}", file=sys.stderr)
        return 2
    records = read_records(args.records, window)
    stored = sum(1 for _ in args.records.glob("*.txt"))
    if stored and not records:
        # For example the records of another preset's build directory: valid, but nothing from this run counts.
        print(f"form_coverage: {args.records}: none of the {stored} records here was written during this JUnit "
              "run, so no record-mode or shader form is counted", file=sys.stderr)
    m = measure(results, records=records, gaps_for=args.gaps)
    if args.gaps:
        gaps = m[args.gaps].pop("gaps")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    if args.gaps:
        print("group\tform\tevidence")
        for row in gaps:
            print("\t".join(row))
    else:
        print(report(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
