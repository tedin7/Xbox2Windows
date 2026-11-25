"""Check heap-view buffer_load_format in the compute store profile: lowering, refusals and readback.

The t0 V# is not static: the format, stride and num_records come from a view record chosen at run time by a
constant word (--heap-view). The records are the side table the runtime fills when it writes a heap view. The
case file supplies the t0 words, so one dispatch sees data, records and the slot chosen by the test.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_load_format_check as fmt  # noqa: E402
import compute_store_check as base  # noqa: E402

aot = base.aot
NAME = "heap_view_load"
GROUP = (4, 1, 1)
THREADS = 8
OUTPUT_STRIDE = 16
WORDS = 320
TABLE = 576                 # byte offset of the view records inside t0
RECORD = aot.HEAP_VIEW_RECORD_BYTES
LIVE = aot.HEAP_VIEW_LIVE
MASK = 0xffffffff
IDENTITY_UNKNOWN = aot.DESCRIPTOR_GFX10_FIXED | (2 << 12)   # 8_SNORM: not handled
# One view: label, format word, base byte, stride, num_records, state. The format code is 0 for a raw word.
# The xy and xyzw fixtures each have a table of twelve views; SLOTS lists the slot each dispatch reads.




def view(label, code, base_byte, stride, records, state=LIVE, legacy=False, word=None):
    """One heap view: the format is a code of the descriptor-format table, or a raw V# word 3."""
    word = aot.descriptor_format_word(code, legacy=legacy) if word is None else word
    return dict(label=label, word=word, base=base_byte, stride=stride, records=records, state=state, code=code)


# opcode width -> (views, source seed). Every dispatch has group x2 = 8 threads reading index = thread.
XY_VIEWS = (
    view("16_16_UNORM, whole range", 23, 0, 16, 8),
    view("32_32_FLOAT, index past num_records", 64, 128, 32, 6),
    view("16_16_FLOAT legacy word, stride 8", 29, 400, 8, 8, legacy=True),
    view("stride 4 holds less than the opcode reads", 23, 0, 4, 8),
    view("view not live (never written)", 23, 0, 16, 8, state=0),
    view("zero stride", 23, 0, 0, 8),
    view("unknown format 8_SNORM", None, 0, 16, 8, word=IDENTITY_UNKNOWN),
    view("four-channel format under an xy opcode", 56, 0, 16, 8),
    view("view end leaves the buffer", 23, 1260, 16, 8),
    view("view released (stale state)", 23, 0, 16, 8, state=2),
    view("base not DWORD aligned", 23, 2, 16, 8),
    view("stride not DWORD aligned", 23, 0, 18, 8),
    view("good view past the declared table count", 23, 0, 16, 8),
)
XYZW_VIEWS = (
    view("8_8_8_8_UNORM, whole range", 56, 0, 16, 8),
    view("32_32_32_32_UINT, index past num_records", 75, 128, 32, 5),
    view("16_16_16_16_FLOAT legacy word, stride 16", 71, 400, 16, 8, legacy=True),
    view("two-channel format under an xyzw opcode", 23, 0, 16, 8),
    view("view not live (never written)", 56, 0, 16, 8, state=0),
    view("zero stride", 56, 0, 0, 8),
    view("unknown format 8_SNORM", None, 0, 16, 8, word=IDENTITY_UNKNOWN),
    view("stride 12 holds less than the opcode reads", 56, 0, 12, 8),
    view("view end leaves the buffer", 56, 1260, 16, 8),
    view("view released (stale state)", 56, 0, 16, 8, state=2),
    view("base not DWORD aligned", 56, 2, 16, 8),
    view("stride not DWORD aligned", 56, 0, 18, 8),
    view("good view past the declared table count", 56, 0, 16, 8),
)
SUFFIXES = {"xy": XY_VIEWS, "xyzw": XYZW_VIEWS}
CHANNELS = {"xy": 2, "xyzw": 4}
# slot -> "ok": results published (zeros where out of range), "refused": u0 untouched.
REFUSED = {"xy": {4, 5, 6, 7, 9, 10, 11, 12}, "xyzw": {3, 4, 5, 6, 9, 10, 11, 12}}
DECLARED = 12               # views the heap_view binding declares; the last record in the table is past it


def source_words(views):
    """Data words below TABLE, then the 32-byte view records. A pseudo-random fill with edge values up front."""
    words, state = [], 0x2545f491
    edge = fmt.CURATED
    for index in range(TABLE // 4):
        if index < len(edge) * (fmt.SOURCE_STRIDE // 4) and (index // 20) < len(edge):
            words.append(edge[index // 20][index % 20])
        else:
            state = (state * 1103515245 + 12345) & MASK
            words.append(state ^ (state >> 13))
    for entry in views:
        words += [entry["base"], entry["stride"] << 16, entry["records"], entry["word"], entry["state"], 0, 0, 0]
    return words + [base.SENTINEL] * (WORDS - len(words))


def reference(entry, words, suffix, thread, *, wrong=()):
    """Result words of one load: the view definition applied to the t0 words, not the generated shader."""
    channels = CHANNELS[suffix]
    stride, records = entry["stride"], entry["records"]
    if "no num_records bound" in wrong:
        records = 1 << 30
    if "fixed stride" in wrong:
        stride = 16
    if thread >= records or 4 * channels > stride:
        return None if "out of range unwritten" in wrong else [0] * channels
    start = entry["base"] + thread * stride
    if start + 4 * channels > WORDS * 4:
        return [0] * channels
    return fmt.load(entry["code"], words[start // 4:start // 4 + channels], channels)


def expected_words(suffix, slot, groups, words, *, wrong=()):
    views = SUFFIXES[suffix]
    result = [base.SENTINEL] * WORDS
    if slot in REFUSED[suffix] or slot >= len(views):
        return result
    for thread in range(GROUP[0] * groups):
        values = reference(views[slot], words, suffix, thread, wrong=wrong)
        if values is None:
            continue
        start = thread * OUTPUT_STRIDE // 4
        result[start:start + len(values)] = values
    return result


def program(suffix):
    channels = CHANNELS[suffix]
    regs = "v10" if channels == 1 else f"v[10:{9 + channels}]"
    width = f"x{channels}" if channels > 1 else ""
    lines = [
        "s_mov_b32 s12, s7", "s_mov_b32 s13, 0",
        "s_load_dwordx4 s[20:23], s[12:13], 0x30",
        "s_load_dwordx4 s[16:19], s[12:13], 0x40",
        "s_load_dwordx4 s[40:43], s[12:13], 0x50",
        "v_lshl_add_u32 v8, s11, 2, v0",
        "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dwordx4 s[24:27], s[16:19], null",
        "s_waitcnt lgkmcnt(0)",
        f"buffer_load_format_{suffix} {regs}, v8, s[40:43], 0 idxen",
        "s_waitcnt vmcnt(0)",
        f"buffer_store_dword{width} {regs}, v8, s[20:23], 0 idxen",
        "s_endpgm",
    ]
    binding = dict(table_reg=7, descriptor_offset=0x30, constant_descriptor=0x40, source_descriptor=0x50,
                   group_reg=11, stride=OUTPUT_STRIDE, cpu="gfx1013", heap_view=(0, TABLE, DECLARED, WORDS * 4))
    return lines, binding


PROGRAMS = {suffix: program(suffix) for suffix in SUFFIXES}


def lower(suffix="xy", lines=None, **binding):
    base_lines, base_binding = PROGRAMS[suffix]
    return aot.lower(base_lines if lines is None else lines, GROUP, **(base_binding | binding))


def constants(slot):
    return [slot] + [0] * 15


def cli(binding):
    flags = base.binding_cli({key: value for key, value in binding.items() if key != "heap_view"})
    return flags + ["--heap-view", *map(str, binding["heap_view"])]


def controls():
    """Exact staged load, bounds and lifetime guards, then one rejection per form outside the profile."""
    for suffix, channels in CHANNELS.items():
        source = lower(suffix)
        lines = source.splitlines()
        call = (f"    uint4 format_load_0 = load_heap_view_format(params[0].x, tid.x, 0u, {channels}u, "
                "format_valid_0);")
        assert call in lines, source
        assert "    if (!(format_valid_0)) return;" in lines
        # The load and the refusal come before the first store.
        store = next(i for i, line in enumerate(lines) if "outputData.Store" in line)
        assert lines.index(call) < lines.index("    if (!(format_valid_0)) return;") < store
        assert "uint4 load_descriptor_format(" in source and "load_descriptor_format(v.w," in source
        assert f"uint record = {TABLE}u + slot * 32u;" in source and "if (slot >= 12u) return zero;" in source and "const uint length = 1280u;" in source
        assert "inputData.Load(record + 16u) != 1u" in source
        assert "0x40000u" not in source and "index <= left / stride" in source
    xy_lines, _ = PROGRAMS["xy"]
    load = "buffer_load_format_xy v[10:11], v8, s[40:43], 0 idxen"
    assert load in xy_lines
    assert "buffer_load_format_xy v[10:11], v8, s[40:43], 0 idxen offset:4" in base.variant(xy_lines, (load, load + " offset:4"))
    # A heap load needs its static counterpart's constant word, a gfx1013 constant buffer and a source binding.
    wait = xy_lines.index("s_waitcnt lgkmcnt(0)", 7)
    for message, lines_, binding in (
        ("Descriptor-format d16 loads are not lowered",
         base.variant(xy_lines, (load, "buffer_load_format_d16_xy v[10:11], v8, s[40:43], 0 idxen")), {}),
        ("Unsupported formatted buffer read target or address mode",
         base.variant(xy_lines, (load, load.replace(" idxen", " offen"))), {}),
        ("Unsupported formatted buffer read target or address mode",
         base.variant(xy_lines, (load, load.replace(" idxen", ""))), {}),
        ("Heap view slot constant is not ready", xy_lines[:wait] + xy_lines[wait + 1:], {}),
        ("Heap view slot constant is not ready", None, dict(heap_view=(15, TABLE, 12, 1280))),
        ("Unused heap view binding", base.variant(xy_lines, (load, "v_mov_b32_e32 v10, 0"),
                                                  ("buffer_store_dwordx2 v[10:11]", "buffer_store_dword v10")), {}),
        ("Invalid heap view binding", None, dict(cpu="gfx1030")),
        ("Invalid heap view binding", None, dict(source_stride=16)),
        ("Invalid heap view binding", None, dict(constant_descriptor=None)),
        ("Invalid buffer load format constant", None, dict(load_format_constant=(0,))),
        ("Invalid heap view binding", None, dict(heap_view=(16, TABLE, 12, 1280))),
        ("Invalid heap view binding", None, dict(heap_view=(0, TABLE + 2, 10, 1280))),
        ("Invalid heap view binding", None, dict(heap_view=(0, TABLE, 0, 1280))),
        ("Invalid heap view binding", None, dict(heap_view=(0, TABLE, aot.HEAP_VIEW_MAX + 1, 1280))),
        ("Invalid heap view binding", None, dict(heap_view=(0, TABLE, 12, 1279))),
        ("Invalid heap view binding", None, dict(heap_view=(0, TABLE, 23, 1280))),
        ("Invalid heap view binding", None, dict(heap_view=(0, TABLE))),
        ("Unsupported store address", base.variant(xy_lines, (", v8, s[20:23], 0 idxen", ", v10, s[20:23], 0 idxen")), {}),
    ):
        base.rejects(message, lower, "xy", lines_, **binding)


def binary(directory, assembler, objcopy, library):
    """LLVM 15 assembles the fixtures; lower of the disassembly and the CLI both match lower of the source."""
    controls()
    for suffix in SUFFIXES:
        lines, binding = PROGRAMS[suffix]
        code = base.assemble(directory, assembler, objcopy, lines, cpu="gfx1013")
        decoded = aot.labelled(base.disassemble(code, library, offsets=True, cpu=b"gfx1013"))
        expected = aot.lower(lines, GROUP, **binding)
        assert aot.lower(decoded, GROUP, **binding) == expected
        container, output = directory / "heap-view.dxbc", directory / "heap-view.hlsl"
        container.write_bytes(base.store_container(code, GROUP))
        subprocess.run([sys.executable, "-B", str(Path(aot.__file__)), str(container), "--output", str(output),
                        "--llvm", library, *cli(binding)], check=True)
        assert output.read_text(encoding="ascii") == expected
    print(f"{NAME}: llvm15_round_trip=match")
    return 0


# label, old, new, the (suffix, slot) run it must fail. Each run is chosen so only that guard decides it.
MUTATIONS = (
    ("num_records ignored", "index < v.z && ", "", ("xy", 1)),
    ("stride read as a fixed 16", "uint stride = (v.y >> 16u) & 0x3fffu;", "uint stride = 16u;", ("xy", 2)),
    ("opcode wider than the stride accepted", " && instOffset + 4u * channels <= stride", "", ("xy", 3)),
    ("liveness not checked", "    if (inputData.Load(record + 16u) != 1u) return zero;\n", "", ("xy", 4)),
    ("stale state read as live", "!= 1u) return zero;", "== 0u) return zero;", ("xy", 9)),
    ("zero stride accepted", "    if (stride == 0u) return zero;\n", "", ("xy", 5)),
    ("buffer end ignored", "inRange = inRange && instOffset + 4u * channels <= left - start;",
     "inRange = inRange;", ("xy", 8)),
    ("index bound ignored", "inRange = index <= left / stride;", "inRange = true;", ("xy", 8)),
    ("slot past the table accepted", "    if (slot >= 12u) return zero;\n", "", ("xy", 12)),
    ("misaligned base or stride accepted", "    if (((v.x | stride | instOffset) & 3u) != 0u) return zero;\n", "", ("xy", 10)),
    ("stride alignment not checked", "(v.x | stride | instOffset)", "(v.x | instOffset)", ("xy", 11)),
    ("base ignored", "addr = v.x + start + instOffset;", "addr = start + instOffset;", ("xy", 1)),
    ("out of range leaves the stale word", "return inRange ? r : zero;", "return r;", ("xy", 1)),
    ("invalid loads still store", "    if (!(format_valid_0)) return;\n", "", ("xy", 6)),
)
WRONG = (("no num_records bound", ("no num_records bound",), ("xy", 1)),
         ("fixed 16-byte stride", ("fixed stride",), ("xy", 2)),
         ("out-of-range threads left unwritten", ("out of range unwritten",), ("xy", 1)))


def draw(directory, compiler, renderer, backend):
    """Dispatch every view; wrong references and mutated sources must fail."""
    cmd = [renderer, backend, "--compute-store-case"]
    shaders = {}
    for suffix in SUFFIXES:
        src, cso = directory / f"heap_{suffix}.hlsl", directory / f"heap_{suffix}.cso"
        src.write_text(lower(suffix), encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
        shaders[suffix] = cso
    case = directory / "heap.case"
    dispatches = 0

    def write(suffix, slot, groups, reference):
        words = source_words(SUFFIXES[suffix])
        case.write_bytes(struct.pack("<321I", groups, *reference) + struct.pack("<16I", *constants(slot))
                         + struct.pack("<320I", *words))
        return words

    for suffix, views in SUFFIXES.items():
        for slot in range(len(views) + 1):
            words = source_words(views)
            for groups in (1, 2):
                write(suffix, slot, groups, expected_words(suffix, slot, groups, words))
                if subprocess.run([*cmd, shaders[suffix], case]).returncode:
                    return 1
                dispatches += 1
            label = views[slot]["label"] if slot < len(views) else "no record"
            print(f"{NAME}: {suffix} slot {slot} {'refused' if slot in REFUSED[suffix] or slot >= len(views) else 'read'}"
                  f" ({label})", flush=True)
    # A refusal must stay distinguishable from a published zero read: zeros against a refused slot must fail.
    for suffix, slot in (("xy", 4), ("xy", 10), ("xy", 11), ("xyzw", 3)):
        words = source_words(SUFFIXES[suffix])
        zeros = [0] * WORDS
        write(suffix, slot, 2, zeros)
        bad = subprocess.run([*cmd, shaders[suffix], case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"{NAME}: zeros published for {suffix} slot {slot} escaped detection\n{bad.stderr}")
    for label, wrong, (suffix, slot) in WRONG:
        words = source_words(SUFFIXES[suffix])
        good = expected_words(suffix, slot, 2, words)
        wrong_words = expected_words(suffix, slot, 2, words, wrong=wrong)
        if good == wrong_words:
            raise AssertionError(f"{NAME}: reference '{label}' matches the expected words")
        write(suffix, slot, 2, wrong_words)
        bad = subprocess.run([*cmd, shaders[suffix], case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"{NAME}: reference '{label}' escaped detection\n{bad.stderr}")
    original = {suffix: lower(suffix) for suffix in SUFFIXES}
    for label, old, new, (suffix, slot) in MUTATIONS:
        mutated = original[suffix].replace(old, new, 1)
        if mutated == original[suffix]:
            raise AssertionError(f"{NAME}: mutation '{label}' did not change the source")
        words = source_words(SUFFIXES[suffix])
        write(suffix, slot, 2, expected_words(suffix, slot, 2, words))
        bad_src, bad_cso = directory / "mutated.hlsl", directory / "mutated.cso"
        bad_src.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", bad_cso, bad_src], check=True)
        bad = subprocess.run([*cmd, bad_cso, case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"{NAME}: mutation '{label}' escaped detection\n{bad.stderr}")
    print(f"{NAME}: {dispatches} dispatches, {3 + len(WRONG)} wrong references and "
          f"{len(MUTATIONS)} mutated sources checked")
    return 0


if __name__ == "__main__":
    mode, *args = sys.argv[1:] or ["controls"]
    controls()
    if mode == "controls":
        sys.exit(0)
    with tempfile.TemporaryDirectory() as tmp:
        if mode == "binary" and len(args) == 3:
            sys.exit(binary(Path(tmp), *args))
        if mode == "draw" and len(args) == 3:
            sys.exit(draw(Path(tmp), *args))
    sys.exit("usage: compute_heap_view_check.py controls | binary <llvm-mc> <llvm-objcopy> <libLLVM> "
             "| draw <dxc> <renderer> <adapter>")
