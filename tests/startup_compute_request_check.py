"""Device slot 67 compute requests on WARP: publishing, refusals and boundaries, and the offline handling.

usage: startup_compute_request_check.py <startup_stream_consumer> [disabled]
       startup_compute_request_check.py <startup_stream_consumer> key-use|key-use-late <dxc>

The consumer creates a WARP device and a root signature, then slot-67 descriptions. The program, data blob and
state blocks are synthetic, and their qwords are not addresses (each is below 0x10000 or at or above 2^47); the
stand-in copies them without interpreting them. Checked:
- req_<key>.c67, .data, .root, .sa, .sb (20 bytes) and .desc (the 12 words with the seven pointer words
  zeroed; words 9 and 10 point into the consumer) hold the exact bytes;
- the key is the runtime's length-framed FNV chain, computed here from the literal in d3d12_draw.inl, equals
  gfx2hlsl.compute_key, and changes when bytes move from one part to the next;
- the second creation and a second process (other addresses) write nothing;
- each refusal (zero or oversize size, a part unreadable to its end next to a PAGE_NOACCESS page, missing root,
  unreadable description) writes no file and logs its reason; code, data and both state blocks at exactly
  their limits publish; valid -> refused -> valid publishes exactly the two valid keys;
- parts left by an interrupted earlier write (no .c67) are kept and the next creation completes the request;
- a data qword that names committed memory is counted and still published;
- gfx2hlsl.py --pending records the compute stage's refusal of the 24-byte state block A in <key>.fail, and --refusals reports compute
  requests in their own section.
"disabled" is the negative control: without X2W_D3D12_RENDER=1 nothing is published, and the check must print
"FAIL: no compute request".
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from startup_stream_check import container, gfx2hlsl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "startup"))
import shader_cache_revision  # noqa: E402

PROGRAM = struct.pack("<2I", 0x7e020280, 0xbf810000)                 # v_mov_b32 v1, 0; s_endpgm
DATA = struct.pack("<2Q", 0xfedcba9876543210, 0xffff00001111008c)     # synthetic, not addresses
STATE = struct.pack("<3Q", 1, 0x2c, 0xfff0000000000003)                # synthetic state block A, 24 bytes
STATE_B = struct.pack("<5I", 0x1234abcd, 0xffff0000, 0, 0, 0)          # the consumer's fixed state block B
RUNTIME = Path(__file__).resolve().parents[1] / "src" / "startup_stubs" / "d3d12_draw.inl"
PARTS = (".c67", ".data", ".root", ".desc", ".sa", ".sb")


def fnv(data, h=1469598103934665603):
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h


def runtime_key(code, data, root, desc, state_a, state_b):
    literal = re.search(rb'constexpr char compute_revision\[\] = "([^"]+)";', RUNTIME.read_bytes())
    graphics = re.search(rb'constexpr char revision\[\] = "([^"]+)";', RUNTIME.read_bytes())
    assert literal and graphics and literal[1] == graphics[1] + b"cs", "the compute revision must follow the graphics one"
    key = fnv(literal[1])
    for part in (code, data, root, desc, state_a, state_b):
        key = fnv(part, fnv(struct.pack("<Q", len(part)), key))
    return key


def run(directory, consumer, env, mode=None, data=DATA, state=STATE, program=PROGRAM):
    (directory / "program").write_bytes(program)
    (directory / "data").write_bytes(data)
    (directory / "state").write_bytes(state)
    args = [consumer, directory / "vertex", directory / "pixel", directory / "root", "slot67",
            directory / "program", directory / "data", directory / "state"] + ([mode] if mode else [])
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    if mode == "key-use-requests" and result.returncode == 77:
        sys.exit(77)
    assert result.returncode == 0, f"consumer exit {result.returncode}\n{result.stderr[-2000:]}"
    return [line for line in result.stderr.splitlines() if "x2w-draw compute request" in line]


def prepare(directory):
    directory.mkdir(exist_ok=True)
    cache = directory / "cache"
    cache.mkdir()
    (directory / "vertex").write_bytes(container(1))
    (directory / "pixel").write_bytes(container(0))
    # The truncated stream that prepares the device publishes nothing, so the cache holds only compute files.
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_RENDER="1", X2W_SHADER_CACHE=str(cache),
               X2W_TEST_TRUNCATED_ONLY="1")
    return cache, env


def names(cache):
    """The request files (revision.txt, which the first request writes, is checked on its own)."""
    return sorted(p.name for p in cache.iterdir() if p.name != shader_cache_revision.REVISION_FILE)


def check_boundaries(directory, consumer):
    refusals = {"code-zero": "code size 0", "data-zero": "data size 0", "no-root": "no root signature blob",
                "code-over": "code size over 1 MiB", "data-over": "data size over 1 MiB",
                "state-over": "state block A size over 4 KiB", "stateb-over": "state block B size over 4 KiB",
                "code-unreadable": "code unreadable", "code-past-end": "code unreadable",
                "data-unreadable": "data unreadable", "state-unreadable": "state block A unreadable",
                "stateb-unreadable": "state block B unreadable", "desc-unreadable": "description unreadable"}
    for mode, reason in refusals.items():
        cache, env = prepare(directory / mode)
        lines = run(directory / mode, consumer, env, mode)
        assert lines == [f"x2w-draw compute request not written: {reason}"], (mode, lines)
        assert names(cache) == [], (mode, names(cache))
    for mode, part, size in (("code-limit", ".c67", 1 << 20), ("data-limit", ".data", 1 << 20),
                             ("state-limit", ".sa", 0x1000), ("stateb-limit", ".sb", 0x1000)):
        cache, env = prepare(directory / mode)
        lines = run(directory / mode, consumer, env, mode)
        assert len(lines) == 1 and " published " in lines[0], (mode, lines)
        published = [p for p in cache.iterdir() if p.suffix == part]
        assert len(published) == 1 and published[0].stat().st_size == size, (mode, names(cache))
    cache, env = prepare(directory / "order")
    lines = run(directory / "order", consumer, env, "order")
    assert [line.split()[4] if "not written" not in line else "refused" for line in lines] == \
        ["published", "refused", "published"], lines
    assert len(names(cache)) == 2 * len(PARTS), names(cache)
    cache, env = prepare(directory / "pointer")
    lines = run(directory / "pointer", consumer, env, "data-pointer")
    assert len(lines) == 1 and " published " in lines[0] and lines[0].endswith("pointer-like blob qwords 1)"), lines
    print("%d refusals write no file; code, data and both state blocks at their limits publish; valid/refused/valid "
          "order; pointer-like data qword counted" % len(refusals))


def check(directory, consumer, disabled):
    cache, env = prepare(directory)
    # A revision.txt left by an older build: the first request replaces it.
    (cache / shader_cache_revision.REVISION_FILE).write_text("graphics x2w-startup-shader-0\n", encoding="ascii")
    if disabled:
        env["X2W_D3D12_RENDER"] = "0"
    lines = run(directory, consumer, env)
    if not list(cache.glob("req_*.c67")):
        print(f"FAIL: no compute request in the cache ({names(cache)}, log {lines})")
        return 1

    root = (directory / "root").read_bytes()
    desc = struct.pack("<12Q", 0, 0, len(STATE), 0, len(STATE_B), 0, len(PROGRAM), 0, len(DATA), 0, 0, 11)
    key = runtime_key(PROGRAM, DATA, root, desc, STATE, STATE_B)
    assert key == gfx2hlsl.compute_key(PROGRAM, DATA, root, desc, STATE, STATE_B), "compute_key differs from the runtime key"
    # Without the length framing, moving four bytes from the data blob to the code would keep the key.
    assert key != gfx2hlsl.compute_key(PROGRAM + DATA[:4], DATA[4:], root, desc, STATE, STATE_B)
    assert gfx2hlsl.fnv(DATA[4:], gfx2hlsl.fnv(PROGRAM + DATA[:4])) == gfx2hlsl.fnv(DATA, gfx2hlsl.fnv(PROGRAM))
    name = "req_%016x" % key
    files = {name + ext: data for ext, data in zip(PARTS, (PROGRAM, DATA, root, desc, STATE, STATE_B))}
    assert names(cache) == sorted(files), names(cache)
    for file, data in files.items():
        assert (cache / file).read_bytes() == data, file
    detail = "(code 8 bytes, data 16 bytes, state blocks 24 + 20 bytes, pointer-like blob qwords 0)"
    assert lines == [f"x2w-draw compute request {key:016x} published {detail}",
                     f"x2w-draw compute request {key:016x} already present {detail}"], lines

    # The stand-in names its revisions next to the request, and they are the ones gfx2hlsl.py at this commit hashes,
    # so the launchers' check (tools/startup/shader_cache_revision.py) stays silent for a matching build.
    revisions = shader_cache_revision.cache_revisions(cache)
    assert revisions == shader_cache_revision.translator_revisions(gfx2hlsl.__file__), revisions
    assert shader_cache_revision.warning(cache, gfx2hlsl.__file__) is None, shader_cache_revision.warning(cache, gfx2hlsl.__file__)
    revision_file = cache / shader_cache_revision.REVISION_FILE

    stamps = {file: (cache / file).stat().st_mtime_ns for file in files}
    revision_stamp = revision_file.stat().st_mtime_ns
    lines = run(directory, consumer, env)
    assert len(lines) == 2 and all("already present" in line for line in lines), lines
    assert {file: (cache / file).stat().st_mtime_ns for file in files} == stamps, "a second process rewrote the request"
    assert revision_file.stat().st_mtime_ns == revision_stamp, "a second process rewrote an unchanged revision.txt"
    assert names(cache) == sorted(files)

    # Each pointed-to part is in the key.
    other_data, other_state = DATA[:-1] + b"\xfe", STATE[:-1] + b"\xfe"
    for data, state in ((other_data, STATE), (DATA, other_state)):
        lines = run(directory, consumer, env, data=data, state=state)
        other = gfx2hlsl.compute_key(PROGRAM, data, root, desc, state, STATE_B)
        assert other != key and lines[0].startswith(f"x2w-draw compute request {other:016x} published"), lines

    # A cache directory that is set but missing: every creation fails and nothing is written anywhere.
    lines = run(directory, consumer, dict(env, X2W_SHADER_CACHE=str(directory / "missing")))
    assert len(lines) == 2 and all(f"{key:016x} not written" in line for line in lines), lines
    assert not (directory / "missing").exists()

    # An interrupted earlier write: the first four parts exist, the root and the marker do not.
    retry, env = prepare(directory / "retry")
    for file, data in files.items():
        if file.endswith((".desc", ".sa", ".sb", ".data")):
            (retry / file).write_bytes(data)
    kept = {file: (retry / file).stat().st_mtime_ns for file in names(retry)}
    lines = run(directory / "retry", consumer, env)
    assert lines[0] == f"x2w-draw compute request {key:016x} published {detail}", lines
    assert names(retry) == sorted(files) and all((retry / f).read_bytes() == d for f, d in files.items())
    assert all((retry / f).stat().st_mtime_ns == t for f, t in kept.items()), "an existing part was rewritten"

    out = subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], check=True,
                         capture_output=True, text=True).stdout
    assert f"{key:016x}: unsupported: compute state block A is 24 bytes, not 28" in out, out
    assert (cache / ("%016x.fail" % key)).read_text() == "compute state block A is 24 bytes, not 28\n"
    assert not (cache / ("%016x.txt" % key)).exists()
    assert all((cache / file).read_bytes() == data for file, data in files.items()), "the offline pass changed the request"
    report = gfx2hlsl.refusals_report(cache)
    assert report["per_request"]["examined"] == 0, report["per_request"]
    assert report["compute"] == dict(examined=3, no_refusal=0, reasons=[
        dict(reason="compute state block A is 24 bytes, not 28", containing=3, first=3)]), report["compute"]

    check_boundaries(directory, consumer)
    print(f"startup_compute_request: PASS on WARP (request {key:016x}: exact files, length-framed runtime key, "
          "published once across two creations and two processes, refusals and limits, interrupted write completed, "
          "offline refusal recorded)")
    return 0


KEY_USE = re.compile(r"^x2w-render key use compute ([0-9a-f]{16}) first ([0-9.]+) s "
                     r"last ([0-9.]+) s uses (\d+) (ok|refused|pending|none)$", re.M)


def census_rows(text):
    rows = {}
    for key, first, last, count, status in KEY_USE.findall(text):
        assert float(first) <= float(last), (key, first, last)
        uses = int(count)
        assert key not in rows or rows[key][0] < uses, (key, text)
        rows[key] = (uses, status)
    return rows


def check_key_use(directory, consumer, dxc, late):
    from startup_compute_dispatch_check import dispatch
    from startup_gfx2hlsl_compute_check import root_blob

    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP="1", X2W_D3D12_DEBUG="1", X2W_NOLOCK_REFUSE="1",
               X2W_COMPUTE_KEY_USE="1", X2W_TEST_COMPUTE_USE_LATE=str(int(late)))
    for name in ("X2W_TEST_DISPATCH_HARDWARE", "X2W_TEST_DISPATCH_IN", "X2W_TEST_DISPATCH_FIELDS"):
        env.pop(name, None)
    root = directory / "compute_root"
    root.write_bytes(root_blob(table_range=True))
    env["X2W_TEST_COMPUTE_ROOT"] = str(root)
    # The slot-67 bytes identify the request. The shader actually dispatched is compiled from the HLSL below.
    program = struct.pack("<I", 0xbf810000)  # s_endpgm
    data = struct.pack("<Q", 0x6d10)
    state = struct.pack("<Q", 0x4321)
    published = run(directory, consumer, env, "key-use-requests", program=program, data=data, state=state)
    keys = [line.split()[3] for line in published if " published " in line]
    assert len(set(keys)) == len(keys) == 3, published
    ok, refused, missing = keys
    (cache / f"{refused}.fail").write_text("synthetic compute refusal\n", encoding="ascii")
    (cache / f"{ok}.txt").write_text("x2w-manifest 2\nstage cs\nud 1\n"
                                    "udsrc 0 rootconst 0 0\nuav 0 table 1 16\n", encoding="ascii")
    shader = directory / "census.hlsl"
    shader.write_text("""
cbuffer Index : register(b0) { uint i; }
RWByteAddressBuffer result : register(u0);
[numthreads(1, 1, 1)]
void main() { result.Store(i * 4, 17 + i * (i + 3)); }
""", encoding="ascii")
    subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", cache / f"{ok}.cs.cso", shader], check=True)
    actual, run_result = dispatch(directory, consumer, env, "dispatch-key-use")
    reference = [0xc5c5c5c5] * len(actual)
    for i in range(7):
        reference[i] = 17 + i * (i + 3)
    assert actual == reference, (actual[:10], reference[:10], run_result.stderr[-3000:])
    assert "key use burst: draw_lock holds +0, untranslated +1000" in run_result.stdout, run_result.stdout
    assert "after pipeline release:" in run_result.stdout, run_result.stdout
    assert "end of initial compute census" in run_result.stderr, run_result.stderr
    initial, heartbeat = run_result.stderr.split("end of initial compute census", 1)
    assert initial.count("ExecuteIndirect refused: no translation") == 2, initial
    before = census_rows(initial)
    after = census_rows(run_result.stdout.split("after pipeline release:", 1)[1])
    expected = {ok: (7, "ok"), refused: (7, "refused"), missing: (3, "none")}
    if late:
        # Require this specific failure after the readback and lock checks pass.
        assert before == after == {ok: (7, "ok")}, (before, after, expected)
        assert before != expected
        print("Late-count mutant rejected: refused and missing dispatches vanished from the census")
    else:
        assert before == after == expected, (before, after, expected, run_result.stderr[-3000:])
        assert census_rows(heartbeat) == {missing: (1003, "none")}, heartbeat
        print("WARP readback matched the CPU loop. Counts 7/7/3 survived Release.")
        print("No draw lock holds in 1,000 refusals. The heartbeat reported only the changed key.")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 3 and args[1] in ("key-use", "key-use-late"):
        with tempfile.TemporaryDirectory() as temporary:
            sys.exit(check_key_use(Path(temporary), args[0], args[2], args[1] == "key-use-late"))
    if len(args) not in (1, 2) or args[1:] not in ([], ["disabled"]):
        sys.exit(__doc__)
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), args[0], args[1:] == ["disabled"]))
