"""The parallel check runner also runs the checks listed in a directory of text files."""
import os
import runpy
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "tools", "run_checks_parallel.py")


def run(*args):
    return subprocess.run([sys.executable, "-B", RUNNER, "-j", "2", *args],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def q(path):
    """Quote a path for a listing line so a directory with spaces stays one word."""
    return '"' + path + '"'


def write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


with tempfile.TemporaryDirectory(prefix="x2w frag ") as tmp:
    echo = os.path.join(tmp, "echo.py")
    write(echo, "import sys\nprint('echo ran', *sys.argv[1:])\n")
    lists = os.path.join(tmp, "lists")
    os.mkdir(lists)

    # Files are read in name order; comments, blank lines and README files are ignored; a line may
    # carry arguments; a command also given on the command line runs once.
    write(os.path.join(lists, "b.txt"), f"# note\n\n{q(echo)} second\n")
    write(os.path.join(lists, "a.txt"), f"{q(echo)} first\n{q(echo)}\n")
    write(os.path.join(lists, "README.md"), f"{q(echo)} never\n")
    ok = run("--from-dir", lists, echo)
    assert ok.returncode == 0, ok.stdout
    assert "all 3 checks passed" in ok.stdout, ok.stdout
    assert ok.stdout.index("echo ran first") < ok.stdout.index("echo ran second"), ok.stdout
    assert "never" not in ok.stdout, ok.stdout

    # The directory alone is enough, and an empty directory is a plain empty run.
    only = run("--from-dir", lists)
    assert only.returncode == 0 and "all 3 checks passed" in only.stdout, only.stdout
    empty = os.path.join(tmp, "empty")
    os.mkdir(empty)
    assert run("--from-dir", empty).returncode == 0

    # A listed failure fails the run like an inline one.
    bad = os.path.join(tmp, "bad.py")
    write(bad, "import sys\nsys.exit(3)\n")
    bad_dir = os.path.join(tmp, "bad_dir")
    os.mkdir(bad_dir)
    write(os.path.join(bad_dir, "x.txt"), q(bad) + "\n")
    failed = run("--from-dir", bad_dir)
    assert failed.returncode == 1 and "FAILED: " + bad in failed.stdout, failed.stdout

    # Loud refusals: missing directory, missing script, duplicate, bad quoting, nothing to run.
    missing_dir = run("--from-dir", os.path.join(tmp, "nope"))
    assert missing_dir.returncode == 1 and "does not exist" in missing_dir.stdout, missing_dir.stdout
    for name, body, needle in (
        ("missing", q(os.path.join(tmp, "gone.py")) + "\n", "x.txt:1"),
        ("dup", f"{q(echo)} a\n{q(echo)} a\n", "duplicate of"),
        ("quote", f"{q(echo)} 'a\n", "x.txt:1"),
    ):
        bad_list = os.path.join(tmp, "case_" + name)
        os.mkdir(bad_list)
        write(os.path.join(bad_list, "x.txt"), body)
        refused = run("--from-dir", bad_list)
        assert refused.returncode == 1 and needle in refused.stdout, (name, refused.stdout)
        assert "echo ran" not in refused.stdout, (name, refused.stdout)
    # The same command in two files is a duplicate too, even with different path spellings.
    two = os.path.join(tmp, "case_two")
    os.mkdir(two)
    write(os.path.join(two, "a.txt"), f"{q(echo)}\n")
    write(os.path.join(two, "b.txt"), f"{q(echo)}\n")
    twice = run("--from-dir", two)
    assert twice.returncode == 1 and "b.txt:1" in twice.stdout and "duplicate" in twice.stdout, twice.stdout
    dot_two = os.path.join(tmp, "case_dot_two")
    os.mkdir(dot_two)
    dot_echo = os.path.join(tmp, ".", os.path.basename(echo))
    write(os.path.join(dot_two, "a.txt"), f"{q(echo)}\n")
    write(os.path.join(dot_two, "b.txt"), f"{q(dot_echo)}\n")
    dot_twice = run("--from-dir", dot_two)
    assert dot_twice.returncode == 1 and "b.txt:1" in dot_twice.stdout and "duplicate" in dot_twice.stdout, dot_twice.stdout
    dedup = run("--from-dir", lists, dot_echo)
    assert dedup.returncode == 0 and "all 3 checks passed" in dedup.stdout, dedup.stdout
    assert run().returncode == 2

    # A listing saved with a UTF-8 byte-order mark (Notepad, Windows PowerShell 5.1 Set-Content
    # -Encoding UTF8) behaves like the same listing without one, also when line 1 is a comment or
    # blank. The mark is written as explicit bytes so the result does not depend on the platform.
    bom = b"\xef\xbb\xbf"
    for name, body in (
        ("path", q(echo) + " bomarg\n"),
        ("comment", "# note\n" + q(echo) + " bomarg\n"),
        ("blank", "\n" + q(echo) + " bomarg\n"),
    ):
        results = []
        for mark in (b"", bom):
            listing = os.path.join(tmp, f"case_bom_{name}_{'bom' if mark else 'plain'}")
            os.mkdir(listing)
            with open(os.path.join(listing, "x.txt"), "wb") as handle:
                handle.write(mark + body.encode("utf-8"))
            results.append(run("--from-dir", listing))
        for result in results:
            assert result.returncode == 0 and "echo ran bomarg" in result.stdout \
                and "all 1 checks passed" in result.stdout, (name, result.stdout)

# On Windows a listed path keeps its backslashes and quoted words lose their quotes; POSIX
# splitting would eat the backslashes (`C:\x2w\a.py` -> `C:x2wa.py`).
split = runpy.run_path(RUNNER)["split_command"]
assert split(r'C:\x2w\tests\a.py "b c" d', windows=True) == [r"C:\x2w\tests\a.py", "b c", "d"]
assert split(r'a.py --llvm="C:\Program Files\LLVM\x.dll" ""', windows=True) == [
    "a.py", r"--llvm=C:\Program Files\LLVM\x.dll", ""]
assert split('a.py --llvm="/opt/my llvm/x.so"', windows=False) == ["a.py", "--llvm=/opt/my llvm/x.so"]
assert split("tests/a.py 'b c' d", windows=False) == ["tests/a.py", "b c", "d"]

# The lists committed to the repository only name scripts that exist, so CI cannot lose one.
for sub in ("python-linux", "pixel-binary"):
    found = subprocess.run([sys.executable, "-B", "-c",
                            "import runpy,sys;r=runpy.run_path(sys.argv[1]);"
                            "print(len(r['read_check_files'](sys.argv[2])))",
                            RUNNER, os.path.join(ROOT, "tests", "checks", sub)],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert found.returncode == 0, found.stdout
print("run_checks_fragments_check ok")
