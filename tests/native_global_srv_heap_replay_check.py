"""GPU-selected global SRV heap slot: native recording and offline request replay (issue #2908).

usage: native_global_srv_heap_replay_check.py FIXTURE COMPILE_DXIL SDK ADAPTER

A source-built compute container picks a descriptor heap slot from a GPU buffer (t0), reads 64 words through
ResourceDescriptorHeap[slot] and stores them to u0. Slots 0 and 1 are raw buffers with different data, slot 2 is a
texture, slot 3 a typed buffer. The fixture's `record` mode captures the GPU selector and the slot view types into a
req_<key>.views file and refuses the incompatible slots before the dispatch; `compile_dxil` and the fixture's `replay`
mode consume the same file offline. Exit 77 when the fixture reports a capability skip.
"""
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

fixture, compile_dxil, sdk, adapter = sys.argv[1:5]
RAW = "global SRV slot {n} has view type {t}, the shader needs raw_buffer"
HEAP_REFUSAL = "direct resource-heap load requires the selected GPU slot and its view type"


def words(*values):
    return struct.pack(f"<{len(values)}I", *values)


def pack(body, extra=()):
    raw = b"FSSM" + struct.pack("<IQ", 0x101, len(body)) + body
    compressed = zlib.compress(raw)
    blob = b"XBCH" + struct.pack("<IQQ", 3, len(compressed), len(raw)) + compressed
    part = bytes(24) + b"XDAT" + words(1, 0, 13, 48, len(blob)) + blob
    chunks = [(b"DXIL", part), *extra]
    table = []
    data = b""
    for name, payload in chunks:
        table.append(32 + 4 * len(chunks) + len(data))
        data += name + words(len(payload)) + payload
        data += bytes(-len(data) % 4)
    return b"DXBC" + bytes(16) + words(1, 32 + 4 * len(chunks) + len(data), len(chunks), *table) + data


def compute_two_inputs(code, threads=64, *, vgprs=4, sa4=0):
    """compute() of native_direct_container_check.py with both root inputs in one list: an SRV/UAV descriptor table
    in s[0:1] (type 3) and the descriptor heap in s[2:3] (type 5). Four user SGPRs: rsrc2 0x8400 | 0x1008."""
    body = bytearray(256)
    struct.pack_into("<Q", body, 0, 6)
    for offset in (8, 16, 24, 32, 40, 48, 72):
        struct.pack_into("<Q", body, offset, 0xDEADBEEFDEADBEEF)
    record = 112
    struct.pack_into("<Q", body, 96, record)
    struct.pack_into("<Q", body, record + 40, len(body))
    struct.pack_into("<I", body, record + 56, 2)
    body.extend(struct.pack("<IIIIBBH", 3, 0, 1, 0, 0, 0, 0))  # A: table of root parameter 0 at user SGPR 0
    body.extend(struct.pack("<IIIIBBH", 5, 0, 1, 0, 2, 0, 0))  # B: descriptor heap at user SGPR 2
    struct.pack_into("<QQ", body, record, len(body), len(code))
    struct.pack_into("<I", body, record + 16, sa4)
    struct.pack_into("<I", body, record + 20, 32)
    struct.pack_into("<II", body, record + 76, 512, vgprs)
    struct.pack_into("<IIII", body, record + 120, 0x8400 | 0x1008, threads, 1, 1)
    body.extend(code)
    return pack(body)


# Assembled with LLVM 15: clang -cc1as -triple amdgcn-amd-amdhsa -target-cpu gfx1013 -show-encoding.
PROGRAM = words(
    0xbe810380,  # s_mov_b32 s1, 0
    0xbe830380,  # s_mov_b32 s3, 0
    0xf4080200, 0xfa000000,  # s_load_dwordx4 s[8:11], s[0:1], 0x0        (t0: the selector)
    0xf4080300, 0xfa000020,  # s_load_dwordx4 s[12:15], s[0:1], 0x20      (u0: the output)
    0xf4200404, 0xfa000000,  # s_buffer_load_dword s16, s[8:11], 0x0      (the selected slot)
    0xbf8cc07f,  # s_waitcnt lgkmcnt(0)
    0x8f118510,  # s_lshl_b32 s17, s16, 5
    0xf4080501, 0x22000000,  # s_load_dwordx4 s[20:23], s[2:3], s17       (heap descriptor of the slot)
    0xbf8cc07f,  # s_waitcnt lgkmcnt(0)
    0x34020082,  # v_lshlrev_b32_e32 v1, 2, v0
    0xe0301000, 0x80050201,  # buffer_load_dword v2, v1, s[20:23], 0 offen
    0xbf8c3f70,  # s_waitcnt vmcnt(0)
    0xe0701000, 0x80030201,  # buffer_store_dword v2, v1, s[12:15], 0 offen
    0xbf810000,  # s_endpgm
)


def key_of(root, cdx):
    """FNV-1a 64 as read_compute_container (src/shader_compute_request.cpp)."""
    key = 1469598103934665603
    for byte in b"native-container-1" + struct.pack("<Q", len(root)) + root + cdx:
        key = ((key ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return key


def run(command):
    result = subprocess.run([str(part) for part in command], capture_output=True, timeout=100)
    out = result.stdout.decode("utf-8", "replace")
    err = result.stderr.decode("utf-8", "replace")
    print("$ " + " ".join(pathlib.Path(str(c)).name if i == 0 else str(c) for i, c in enumerate(command)))
    for text in (out, err):
        for line in text.strip().splitlines():
            print("    " + line)
    print(f"    -> exit {result.returncode}")
    if result.returncode == 77 and str(command[0]) == fixture:
        print("fixture capability skip (77)")
        sys.exit(77)
    return result.returncode, out, err


def require(condition, message):
    if not condition:
        print("FAIL: " + message)
        sys.exit(1)
    print("ok: " + message)


with tempfile.TemporaryDirectory(prefix="x2w_global_srv_") as scratch:
    base = pathlib.Path(scratch)
    cache = base / "cache"
    cache.mkdir()

    # Fixture root signature, container, key (B6).
    root_path = base / "fixture.root"
    code, out, err = run([fixture, "root", root_path])
    require(code == 0 and root_path.stat().st_size > 0, "fixture wrote the root signature")
    root = root_path.read_bytes()
    cdx = compute_two_inputs(PROGRAM)
    stem = "req_%016x" % key_of(root, cdx)

    def request_dir(name):
        directory = base / name
        directory.mkdir()
        (directory / (stem + ".cdx")).write_bytes(cdx)
        (directory / (stem + ".root")).write_bytes(root)
        return directory

    def compile_directory(directory):
        out_dir = base / (directory.name + "_out")
        result = run([compile_dxil, sdk, cache, "gfx1013-container-cs", "--requests", directory, out_dir])
        # compile_dxil --requests publishes <key>.cs.cso (the req_ prefix stripped) for a compute request.
        return result, out_dir / (stem[len("req_"):] + ".cs.cso")

    selections = [request_dir(f"sel{n}") for n in range(4)]
    nometa = request_dir("nometa")

    # 1. Missing metadata, offline: a heap shader without .views is refused before any output exists.
    (code, out, err), cso = compile_directory(nometa)
    require(code == 1 and HEAP_REFUSAL in err, "compile_dxil without .views refuses the heap read with the slot/type text")
    require(not cso.exists(), "no .cso is published for the request without .views")

    # 2. Native recording path: accepted and refused selections.
    for n in (0, 1):
        views = selections[n] / (stem + ".views")
        code, out, err = run([fixture, "record", sdk, cache, selections[n] / (stem + ".cdx"), adapter, n, "ok"])
        require(code == 0, f"record select {n}: fenced output equals the slot {n} data and u0 sentinels are intact")
        require("output matches slot %d data" % n in out, f"record select {n} compared 64 words with the CPU reference")
        text = views.read_text()
        require(text.startswith("x2w-global-srv-views 2\nselect %d\nneed raw_buffer\n" % n),
                f"record select {n} captured the GPU selector and the view type")
        for line in ("slot 0 raw_buffer 1 256", "slot 1 raw_buffer 1 256", "slot 2 texture 0 0", "slot 3 typed_buffer 4 256"):
            require(line + "\n" in text, f"captured view line '{line}'")
    for n, type_name in ((2, "texture"), (3, "typed_buffer")):
        code, out, err = run([fixture, "record", sdk, cache, selections[n] / (stem + ".cdx"), adapter, n, "refused"])
        require(code == 0, f"record select {n}: refused with unchanged u0 sentinels")
        require(RAW.format(n=n, t=type_name) in out, f"record select {n}: slot/type diagnostic before the dispatch")
        require("no dispatch submitted" in out and "sentinels intact" in out, f"record select {n}: nothing was dispatched")

    # 3. Discrimination control: without the guard a valid slot runs and changes u0, which the check must see.
    code, out, err = run([fixture, "record", sdk, cache, selections[0] / (stem + ".cdx"), adapter, 0, "refused", "--no-guard"])
    require(code == 3 and "sentinels changed" in out, "record --no-guard expecting a refusal sees the changed sentinels (exit 3)")

    # 4. Offline replay compilation of the captured requests.
    outputs = {}
    for n in range(4):
        (code, out, err), cso = compile_directory(selections[n])
        outputs[n] = cso
        if n < 2:
            require(code == 0 and "requests=1 compiled=1 refused=0" in out, f"compile_dxil select {n}: compiled")
            require("global_srv select=%d need=raw_buffer" % n in out, f"compile_dxil select {n}: reports the captured selection")
            require(cso.exists(), f"compile_dxil select {n}: published a .cso")
        else:
            type_name = "texture" if n == 2 else "typed_buffer"
            require(code == 1 and "refused=1" in out, f"compile_dxil select {n}: refused")
            require(RAW.format(n=n, t=type_name) in err, f"compile_dxil select {n}: slot/type diagnostic")
            require(not cso.exists(), f"compile_dxil select {n}: no .cso")

    # 5. Fixture replay of the compiled shader and of the refusals.
    def replay(directory, cso, views, expect, reference):
        return run([fixture, "replay", sdk, cache, directory / (stem + ".cdx"), cso, views, adapter, expect, reference])

    for n in (0, 1):
        code, out, err = replay(selections[n], outputs[n], selections[n] / (stem + ".views"), "ok", n)
        require(code == 0 and "output matches slot %d data" % n in out, f"replay select {n}: compiled shader output equals slot {n} data")
    for n, type_name in ((2, "texture"), (3, "typed_buffer")):
        code, out, err = replay(selections[0], outputs[0], selections[n] / (stem + ".views"), "refused", 0)
        require(code == 0 and RAW.format(n=n, t=type_name) in out and "sentinels intact" in out,
                f"replay select {n}: refused with the slot/type diagnostic, u0 sentinels intact")

    # 6. Mutation control: the slot 1 data is not the slot 0 reference.
    code, out, err = replay(selections[0], outputs[0], selections[0] / (stem + ".views"), "ok", 1)
    require(code == 1 and "mismatch word 0" in out, "replay select 0 compared with slot 1 data reports a mismatch (exit 1)")

    # 7. Negative controls on both paths: metadata that cannot reconstruct the binding.
    good = (selections[0] / (stem + ".views")).read_text()
    lines = good.splitlines()

    def variant(name, text, expected):
        directory = request_dir("neg_" + name)
        views = directory / (stem + ".views")
        views.write_text(text, newline="\n")
        (code, out, err), cso = compile_directory(directory)
        require(code == 1 and expected in err, f"compile_dxil '{name}': refused with '{expected}'")
        require(not cso.exists() and not any(cso.parent.glob("*.cso")), f"compile_dxil '{name}': no .cso")
        code, out, err = replay(selections[0], outputs[0], views, "refused", 0)
        require(code == 0 and expected in out and "sentinels intact" in out,
                f"replay '{name}': refused with the same diagnostic, u0 sentinels intact")

    def without(prefix):
        kept = [line for line in lines if not line.startswith(prefix)]
        require(len(kept) == len(lines) - 1, f"control input drops exactly one '{prefix}' line")
        return "\n".join(kept) + "\n"

    variant("mismatched_type", good.replace("slot 0 raw_buffer 1 256", "slot 0 texture 0 0"), RAW.format(n=0, t="texture"))
    require("slot 0 texture 0 0" in (base / "neg_mismatched_type" / (stem + ".views")).read_text(), "mismatch input rewrote slot 0")
    variant("no_need", without("need "), "global SRV view metadata lacks the needed view type")
    variant("no_select", without("select "), "global SRV view metadata lacks the selected slot")
    variant("no_slot_line", without("slot 0 "), "global SRV slot 0 has no captured view type")
    variant("select_7", good.replace("select 0", "select 7"), "global SRV slot 7 has no captured view type")

    # 8. Shader shapes whose heap read could address another slot than the captured selection, use the slot's stride
    #    as a number, or rewrite the heap descriptor in a way the raw slot view cannot stand for: refused by
    #    compile_dxil even with valid metadata, no .cso.
    words_list = list(struct.unpack(f"<{len(PROGRAM) // 4}I", PROGRAM))
    lshl = words_list.index(0x8f118510)
    heap_load = words_list.index(0xf4080501)
    after_wait = heap_load + 3  # the s_waitcnt after the heap descriptor load
    # The byte-address rewrite of the heap descriptor s[20:23] into s[24:27] (LLVM 15 encodings, as above).
    rewrite = [
        0x939cff15, 0x000e0010,  # s_bfe_u32 s28, s21, 0xe0010          (stride)
        0x849c811c,              # s_max_u32 s28, s28, 1
        0x931a1c16,              # s_mul_i32 s26, s22, s28              (byte count)
        0x8719ff15, 0xc000ffff,  # s_and_b32 s25, s21, 0xc000ffff       (stride cleared)
        0x871bff17, 0xcfffffff,  # s_and_b32 s27, s23, 0xcfffffff
        0x881bff1b, 0x30000000,  # s_or_b32 s27, s27, 0x30000000        (raw out-of-bounds mode)
        0xbe980314,              # s_mov_b32 s24, s20
    ]
    load = words_list.index(0xe0301000)  # buffer_load_dword v2, v1, s[20:23], 0 offen
    store = words_list.index(0xe0701000)  # buffer_store_dword v2, v1, s[12:15], 0 offen

    def window(rewritten=rewrite, prefix=(), store_word=0x80030201, load_words=(0xe0301000, 0x80060201)):
        """PROGRAM with the rewrite after the heap load and the load through s[24:27]."""
        program = list(words_list)
        program[load:load + 2] = load_words  # default: buffer_load_dword v2, v1, s[24:27], 0 offen
        program[store + 1] = store_word
        return program[:after_wait] + list(prefix) + list(rewritten) + program[after_wait:]

    WINDOW = window()
    # tbuffer_load_format_x v2, v1, s[24:27], 0 format:[BUF_FMT_32_UINT] offen: a formatted load through the window.
    # 32_UINT converts nothing, so the output is the slot data as for WINDOW.
    TBUFFER_LOAD = (0xe8a01000, 0x80060201)
    TWINDOW = window(load_words=TBUFFER_LOAD)
    second_load = window([0xbe980320 if w == 0xbe980314 else w for w in rewrite])
    second_load[after_wait - 1:after_wait - 1] = [0xf4080801, 0x22000000]  # before the s_waitcnt
    # The branch word sits at after_wait + 1; its target is next pc + 4 * simm16, here the s_lshl_b32 at word lshl.
    loop_around_load = window(prefix=(0xbf071010, 0xbf850000 | ((lshl - (after_wait + 2)) & 0xFFFF)))
    shapes = {
        # s_mov_b32 s17, 64 instead of the GPU-derived s_lshl_b32 s17, s16, 5
        "constant_index": (words_list[:lshl] + [0xbe9103c0] + words_list[lshl + 1:],
                           "heap descriptor index is a constant; the captured selection must come from GPU data"),
        # s_load_dwordx4 s[20:23], s[2:3], s17 offset:0x20
        "immediate_offset": (words_list[:heap_load] + [0xf4080501, 0x22000020] + words_list[heap_load + 2:],
                             "heap descriptor load with an immediate offset addresses another slot than the captured selection"),
        # s_bfe_u32 s24, s21, 0xe0010 after the heap descriptor load (stride field of the selected slot), then
        # v_mov_b32 v3, s24: the stride is used as a number
        "stride_read": (words_list[:after_wait] + [0x9398ff15, 0x000e0010, 0x7e060218] + words_list[after_wait:],
                        "unbound numeric SGPR s24"),
        # s_and_b32 s25, s21, 0xc000fffe: not the mask that clears the stride
        "window_wrong_mask": (window([0xc000fffe if w == 0xc000ffff else w for w in rewrite]), "unbound numeric SGPR s21"),
        # a second heap load s_load_dwordx4 s[32:35], s[2:3], s17 and word 0 from it (s_mov_b32 s24, s32)
        "window_second_load": (second_load, "non-contiguous resource descriptor"),
        # buffer_store_dword v2, v1, s[24:27], 0 offen: a store through the window
        "window_store": (window(store_word=0x80060201), "a GPU-selected heap slot is read as a raw buffer SRV only"),
        # s_cmp_lg_u32 s16, s16 and s_cbranch_scc1 back to the s_lshl_b32 before the heap load (never taken): the
        # loop contains the heap load, which would run again for another selection
        "window_loop": (loop_around_load, "heap byte window loaded inside a loop"),
        # the same loop around the heap load, read by the formatted load
        "twindow_loop": (window(prefix=loop_around_load[after_wait:after_wait + 2], load_words=TBUFFER_LOAD),
                         "heap byte window loaded inside a loop"),
    }
    for name, (program, expected) in shapes.items():
        directory = base / ("shape_" + name)
        directory.mkdir()
        shape_cdx = compute_two_inputs(words(*program))
        shape_stem = "req_%016x" % key_of(root, shape_cdx)
        (directory / (shape_stem + ".cdx")).write_bytes(shape_cdx)
        (directory / (shape_stem + ".root")).write_bytes(root)
        (directory / (shape_stem + ".views")).write_text(good, newline="\n")
        out_dir = base / (directory.name + "_out")
        code, out, err = run([compile_dxil, sdk, cache, "gfx1013-container-cs", "--requests", directory, out_dir])
        require(code == 1 and expected in err, f"compile_dxil shape '{name}': refused with '{expected}'")
        require(not any(out_dir.glob("*.cso")), f"compile_dxil shape '{name}': no .cso")

    # 9. The byte-address rewrite of the selected heap descriptor (stride, max 1, byte count, stride cleared, raw
    #    out-of-bounds mode) reads the selected raw slot: recorded, compiled offline and replayed like PROGRAM. A stride
    #    read that nothing uses compiles too. The descriptor words stay symbolic, so a stride > 1 is not exercised.
    def shader_request(name, program, **stage):
        directory = base / name
        directory.mkdir()
        cdx_bytes = compute_two_inputs(words(*program), **stage)
        window_stem = "req_%016x" % key_of(root, cdx_bytes)
        (directory / (window_stem + ".cdx")).write_bytes(cdx_bytes)
        (directory / (window_stem + ".root")).write_bytes(root)
        return directory, window_stem

    compiled_shapes = {
        "stride_unused": (words_list[:after_wait] + [0x9398ff15, 0x000e0010] + words_list[after_wait:],
                          "an unused stride read compiles and publishes a .cso"),
        # s_cmp_lg_u32 s16, s16 and s_cbranch_scc1 -2 (back to the compare, never taken) after the heap load: the loop
        # does not contain the load, which runs once
        "window_loop_after_load": (window(prefix=(0xbf071010, 0xbf85fffe)),
                                   "a loop after the heap load does not refuse the window, a .cso is published"),
        # tbuffer_load_format_xyzw v[2:5], v1, s[24:27], 0 format:[BUF_FMT_8_8_8_8_UNORM] offen: a converting format
        # through the window (values through a window are checked by native_mtbuf_check.py on a table window)
        "twindow_unorm": (window(load_words=(0xe9c31000, 0x80060201)),
                          "a converting formatted load through the window compiles and publishes a .cso"),
    }
    for name, (program, message) in compiled_shapes.items():
        directory, window_stem = shader_request(name, program)
        (directory / (window_stem + ".views")).write_text(good, newline="\n")
        out_dir = base / (name + "_out")
        code, out, err = run([compile_dxil, sdk, cache, "gfx1013-container-cs", "--requests", directory, out_dir])
        require(code == 0 and "requests=1 compiled=1 refused=0" in out and
                (out_dir / (window_stem[len("req_"):] + ".cs.cso")).exists(), "compile_dxil: " + message)
    # The raw load (window) and the formatted load (twindow) through the rewritten descriptor.
    for label, program in (("window", WINDOW), ("twindow", TWINDOW)):
        window_outputs = {}
        for n in (0, 1):
            directory, window_stem = shader_request(f"{label}{n}", program)
            code, out, err = run([fixture, "record", sdk, cache, directory / (window_stem + ".cdx"), adapter, n, "ok"])
            require(code == 0 and "output matches slot %d data" % n in out,
                    f"record {label} select {n}: u0 equals the slot {n} data read through the rewritten descriptor")
            out_dir = base / f"{label}{n}_out"
            code, out, err = run([compile_dxil, sdk, cache, "gfx1013-container-cs", "--requests", directory, out_dir])
            cso = out_dir / (window_stem[len("req_"):] + ".cs.cso")
            require(code == 0 and "requests=1 compiled=1 refused=0" in out and cso.exists(),
                    f"compile_dxil {label} select {n}: compiled and published a .cso")
            window_outputs[n] = (directory, window_stem, cso)
            code, out, err = run([fixture, "replay", sdk, cache, directory / (window_stem + ".cdx"), cso,
                                  directory / (window_stem + ".views"), adapter, "ok", n])
            require(code == 0 and "output matches slot %d data" % n in out,
                    f"replay {label} select {n}: output equals slot {n} data")
        directory, window_stem, cso = window_outputs[0]
        code, out, err = run([fixture, "replay", sdk, cache, directory / (window_stem + ".cdx"), cso,
                              directory / (window_stem + ".views"), adapter, "ok", 1])
        require(code == 1 and "mismatch word 0" in out,
                f"replay {label} select 0 compared with slot 1 data reports a mismatch")

    missing = base / "missing" / (stem + ".views")
    code, out, err = replay(selections[0], outputs[0], missing, "refused", 0)
    require(code == 0 and "global SRV view metadata file missing: " + str(missing) in out and "sentinels intact" in out,
            "replay with a missing .views path refuses with the file diagnostic, u0 sentinels intact")

    # 10. Raw load address forms. Eight active lanes each write one 16-byte output slot.
    def raw_load_request(name, width=1, *, offen=False, soffset=18, imm=0, flags=0, flags2=0):
        opcode_delta = {1: 0, 2: 1, 3: 3, 4: 2}[width] << 18
        program = words_list[:load - 1] + [
            0x34020084,              # v_lshlrev_b32_e32 v1, 4, v0
            0xbe92038c,              # s_mov_b32 s18, 12
            0xbefe03ff, 0x000000ff,  # s_mov_b32 exec_lo, 0xff
            0xe0300000 | opcode_delta | (int(offen) << 12) | imm | flags,
            (soffset << 24) | 0x00050201 | flags2,  # s[20:23], vdata at v2, VADDR field v1
            0xbf8c3f70,              # s_waitcnt vmcnt(0)
            0xe0701000 | opcode_delta, 0x80030201,  # buffer_store_dword[xN] v2..., v1, s[12:15], 0 offen
            0xbf810000,              # s_endpgm
        ]
        directory, key = shader_request(name, program, threads=32, vgprs=6, sa4=1 << 20)
        views = directory / (key + ".views")
        views.write_text(good, newline="\n")
        out_dir = base / (name + "_out")
        result = run([compile_dxil, sdk, cache, "gfx1013-container-cs", "--requests", directory, out_dir])
        return result, directory, key, views, out_dir / (key[len("req_"):] + ".cs.cso")

    cases = [(f"off_sgpr_x{width}", width, 12, {}) for width in (1, 2, 3, 4)]
    cases += [
        ("off_sgpr_imm", 1, 12, {"imm": 8}),
        ("offen_sgpr", 1, 12, {"offen": True}),
        ("off_inline", 1, 20, {"soffset": 128 + 20}),
        ("off_inline_zero", 1, 0, {"soffset": 128}),
    ]
    for name, width, scalar_bytes, addressing in cases:
        (code, out, err), directory, key, views, cso = raw_load_request(name, width, **addressing)
        require(code == 0 and "requests=1 compiled=1 refused=0" in out and cso.exists(), f"{name}: compiled")
        expected = [0x69696969] * 72
        for lane in range(8):
            addr = scalar_bytes + addressing.get("imm", 0) + (16 * lane if addressing.get("offen") else 0)
            expected[4 * lane:4 * lane + width] = [0xA0000000 + addr // 4 + k for k in range(width)]
        reference = directory / "expected.bin"
        reference.write_bytes(words(*expected))
        code, out, err = run([fixture, "replay", sdk, cache, directory / (key + ".cdx"), cso,
                              views, adapter, "ok", 0, reference])
        require(code == 0 and "output matches CPU reference (72 words)" in out, f"{name}: CPU reference")

    modifiers = "raw buffer load needs a linear byte offset and no modifiers"
    for name, diagnostic, addressing in (
        ("raw_glc", modifiers, {"flags": 1 << 14}),
        ("raw_slc", modifiers, {"flags2": 1 << 22}),
        ("raw_idxen", modifiers, {"flags": 1 << 13}),
        ("raw_descriptor_soffset", "unbound numeric SGPR s20", {"soffset": 20}),
        ("raw_literal_soffset", "raw buffer load scalar offset needs an SGPR or inline integer", {"soffset": 255}),
    ):
        (code, out, err), _, _, _, cso = raw_load_request(name, **addressing)
        require(code == 1 and diagnostic in err and not cso.exists(), f"{name}: refused with '{diagnostic}'")

    # 11. Typed heap load (HeapViewContract::typed_dword): buffer_load_format_x v2, v0, s[20:23], 0 idxen reads element
    #     `lane` of the selected slot, a one-channel 32-bit typed buffer (slot 3, R32_UINT), so u0 equals the slot 3 data.
    TYPED_RAW = "global SRV slot {n} has view type {t}, the shader needs typed_buffer"
    TYPED_ONLY = "a GPU-selected heap slot is read as a typed dword buffer SRV only"

    def typed_program(load_words):
        program = list(words_list)
        program[load:load + 2] = load_words
        return program

    TYPED = typed_program((0xe0002000, 0x80050200))  # buffer_load_format_x v2, v0, s[20:23], 0 idxen

    def typed_dir(name, program=TYPED):
        directory = base / name
        directory.mkdir()
        cdx_bytes = compute_two_inputs(words(*program))
        typed_stem = "req_%016x" % key_of(root, cdx_bytes)
        (directory / (typed_stem + ".cdx")).write_bytes(cdx_bytes)
        (directory / (typed_stem + ".root")).write_bytes(root)
        return directory, typed_stem

    def typed_compile(directory, typed_stem, out_name):
        out_dir = base / out_name
        code, out, err = run([compile_dxil, sdk, cache, "gfx1013-container-cs", "--requests", directory, out_dir])
        return code, out, err, out_dir / (typed_stem[len("req_"):] + ".cs.cso")

    typed_selection = {}
    for n in (3, 0, 2):
        directory, typed_stem = typed_dir(f"typed{n}")
        typed_selection[n] = (directory, typed_stem)
        expect = "ok" if n == 3 else "refused"
        code, out, err = run([fixture, "record", sdk, cache, directory / (typed_stem + ".cdx"), adapter, n, expect,
                              "--need", "typed_buffer"])
        if n == 3:
            require(code == 0 and "output matches slot 3 data" in out,
                    "record typed select 3: u0 equals the slot 3 data read by the typed heap load")
            text = (directory / (typed_stem + ".views")).read_text()
            require(text.startswith("x2w-global-srv-views 2\nselect 3\nneed typed_buffer\n") and "dword 3 1\n" in text,
                    "record typed select 3 captured need typed_buffer and the slot 3 channel count")
        else:
            type_name = "raw_buffer" if n == 0 else "texture"
            require(code == 0 and TYPED_RAW.format(n=n, t=type_name) in out and "no dispatch submitted" in out and
                    "sentinels intact" in out, f"record typed select {n}: refused before the dispatch")
    directory, typed_stem = typed_selection[3]
    typed_views = (directory / (typed_stem + ".views")).read_text()
    code, out, err, typed_cso = typed_compile(directory, typed_stem, "typed3_out")
    require(code == 0 and "requests=1 compiled=1 refused=0" in out and "global_srv select=3 need=typed_buffer" in out and
            typed_cso.exists(), "compile_dxil typed select 3: compiled and published a .cso")
    code, out, err, again = typed_compile(directory, typed_stem, "typed3_again")
    require(code == 0 and "frontend_cache_hit=1" in out and again.read_bytes() == typed_cso.read_bytes(),
            "compile_dxil typed select 3 again: frontend cache hit (the typed heap binding decodes), same .cso")
    views_path = directory / (typed_stem + ".views")
    code, out, err = run([fixture, "replay", sdk, cache, directory / (typed_stem + ".cdx"), typed_cso, views_path,
                          adapter, "ok", 3])
    require(code == 0 and "output matches slot 3 data" in out, "replay typed select 3: output equals slot 3 data")
    code, out, err = run([fixture, "replay", sdk, cache, directory / (typed_stem + ".cdx"), typed_cso, views_path,
                          adapter, "ok", 0])
    require(code == 1 and "mismatch word 0" in out, "replay typed select 3 compared with slot 0 data reports a mismatch")

    # Refused shapes and metadata, compile_dxil, no .cso.
    typed_negatives = {
        # buffer_load_format_xy v[2:3], v0, s[20:23], 0 idxen
        "typed_xy": (typed_program((0xe0042000, 0x80050200)), typed_views, "typed heap load reads one channel only"),
        "typed_raw_program": (list(words_list), typed_views, TYPED_ONLY),
        # tbuffer_load_format_x v2, v1, s[20:23], 0 format:[BUF_FMT_32_UINT] offen through the heap descriptor
        "typed_tbuffer": (typed_program((0xe8a01000, 0x80050201)), typed_views, TYPED_ONLY),
        "typed_raw_views": (TYPED, good, "a GPU-selected heap slot is read as a raw buffer SRV only"),
        "typed_no_dword": (TYPED, typed_views.replace("dword 3 1\n", ""),
                           "GPU-selected heap slot 3 is not a one-channel 32-bit typed buffer (no 32-bit channel metadata)"),
        "typed_select_0": (TYPED, typed_views.replace("select 3", "select 0"), TYPED_RAW.format(n=0, t="raw_buffer")),
        "typed_dword_5": (TYPED, typed_views.replace("dword 3 1", "dword 3 5"),
                          "global SRV view metadata: unreadable dword line"),
        "typed_dword_raw_slot": (TYPED, typed_views.replace("dword 3 1", "dword 0 1"),
                                 "global SRV view metadata: unreadable dword line"),
        "typed_dword_twice": (TYPED, typed_views + "dword 3 1\n", "global SRV view metadata: unreadable dword line"),
        "typed_dword_first": (TYPED, typed_views.replace("dword 3 1\n", "").replace("need typed_buffer\n",
                              "need typed_buffer\ndword 3 1\n"), "global SRV view metadata: unreadable dword line"),
    }
    for name, (program, views_text, expected) in typed_negatives.items():
        directory, typed_stem = typed_dir("neg_" + name, program)
        (directory / (typed_stem + ".views")).write_text(views_text, newline="\n")
        code, out, err, cso = typed_compile(directory, typed_stem, "neg_" + name + "_out")
        require(code == 1 and expected in err, f"compile_dxil '{name}': refused with '{expected}'")
        require(not any(cso.parent.glob("*.cso")), f"compile_dxil '{name}': no .cso")

print("native_global_srv_heap_replay: all checks passed")
