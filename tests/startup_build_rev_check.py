"""The build revision in the presenter window title and the device-create log line.

  (no arguments)                       run cmake/x2w_build_rev.cmake against temporary git repositories
  dll <d3d12_xs.dll> <x2w_build_rev.cpp>
                                       check that the built shim holds the generated revision and the
                                       title and log formats that print it

The generator writes `git rev-parse --short=8 HEAD`, "-dirty" for tracked changes only, or "unknown" when
git is missing or the source directory is not the top of a work tree. It must leave the file untouched
when the revision is unchanged, so an unchanged tree recompiles nothing.

The controls need git and cmake on PATH. Without them the check exits 77; that is a skip, not a pass.
The repositories use an empty global config, no system config and a test identity, so the developer's
identity, hooks and signing settings never reach them.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "cmake" / "x2w_build_rev.cmake"
REV_LINE = re.compile(r'^extern const char x2w_build_rev\[\] = "([^"]*)";$', re.M)
WIDE_LINE = re.compile(r'^extern const wchar_t x2w_build_rev_w\[\] = L"([^"]*)";$', re.M)
SHAPE = re.compile(r"^(?:[0-9a-f]{8,40}(?:-dirty)?|unknown)$")
OLD_NS = 10**18  # 2001-09-09: a fixed mtime that no write by the generator can produce

# Format strings that print the revision (d3d12_present.inl, d3d12_xs_device.cpp).
TITLE_FORMATS = {
    "output-window title": "Xbox2Windows %ls - output",
    "frame title": "Xbox2Windows %ls - frame %ld, %.1f fps, %ld translated draws "
                   "(title bar is diagnostics, picture is game output)",
}
LOG_FORMAT = "x2w-d3d12 D3D12XboxCreateDevice adapter=%p params=[%s] riid=%s -> 0x%08lx build=%s"


def fail(message):
    print(f"FAIL {message}")
    sys.exit(1)


def read_rev(path, case=None):
    where = f"{case}: {path}" if case else str(path)
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        fail(f"cannot read {where}: {error} (revision unmeasured)")
    narrow, wide = REV_LINE.search(text), WIDE_LINE.search(text)
    if not narrow or not wide:
        fail(f"{where} defines no x2w_build_rev / x2w_build_rev_w (revision unmeasured)")
    if narrow.group(1) != wide.group(1):
        fail(f"{where}: narrow {narrow.group(1)!r} and wide {wide.group(1)!r} revisions differ")
    if not SHAPE.match(narrow.group(1)):
        fail(f"{where}: revision {narrow.group(1)!r} is not <hex>[-dirty] or unknown")
    return narrow.group(1)


class Repo:
    def __init__(self, base, git, cmake):
        self.git, self.cmake = git, cmake
        empty = base / "empty.gitconfig"
        empty.write_text("")
        # An empty hooks folder rather than "core.hooksPath=": some git builds reject an empty path.
        self.hooks = base / "no-hooks"
        self.hooks.mkdir(exist_ok=True)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.env.update(GIT_CONFIG_GLOBAL=str(empty), GIT_CONFIG_NOSYSTEM="1")

    def run(self, cwd, *args):
        result = subprocess.run([self.git, "-c", "user.name=x2w-test", "-c", "user.email=test@example.invalid",
                                 "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={self.hooks.as_posix()}", *args],
                                cwd=cwd, env=self.env, capture_output=True, text=True)
        if result.returncode != 0:
            fail(f"git {' '.join(args)} in {cwd}: {result.stderr.strip()}")
        return result.stdout.strip()

    def generate(self, case, source, output, git=None, env=None):
        result = subprocess.run([self.cmake, f"-DSOURCE_DIR={source}", f"-DOUTPUT={output}",
                                 f"-DGIT_EXECUTABLE={self.git if git is None else git}", "-P", str(SCRIPT)],
                                env={**self.env, **(env or {})}, capture_output=True, text=True)
        if result.returncode != 0:
            fail(f"{case}: generator exited {result.returncode}: {result.stderr.strip()}")
        return read_rev(output, case)


def expect(case, got, want):
    if got != want:
        fail(f"{case}: revision {got!r}, expected {want!r}")
    print(f"ok {case}: {got}")


def controls():
    git, cmake = shutil.which("git"), shutil.which("cmake")
    if not git or not cmake:
        print(f"SKIP git={git} cmake={cmake}: the generator controls need both on PATH")
        sys.exit(77)
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp).resolve()
        repo = Repo(base, git, cmake)
        tree, output = base / "tree", base / "gen" / "x2w_build_rev.cpp"
        tree.mkdir()
        repo.run(tree, "-c", "init.defaultBranch=master", "init", "-q")
        (tree / "tracked.txt").write_text("one\n")
        repo.run(tree, "add", "tracked.txt")
        repo.run(tree, "commit", "-q", "-m", "first")
        first = repo.run(tree, "rev-parse", "--short=8", "HEAD")
        if not re.fullmatch(r"[0-9a-f]{8,40}", first):
            fail(f"git rev-parse gave {first!r}")

        expect("clean commit", repo.generate("clean commit", tree, output), first)

        os.utime(output, ns=(OLD_NS, OLD_NS))
        expect("rerun unchanged", repo.generate("rerun unchanged", tree, output), first)
        if output.stat().st_mtime_ns != OLD_NS:
            fail(f"rerun unchanged: the generator rewrote an unchanged file (mtime {output.stat().st_mtime_ns})")
        print("ok rerun unchanged: file not rewritten")

        (tree / "untracked.txt").write_text("not added\n")
        expect("untracked file only", repo.generate("untracked file only", tree, output), first)

        (tree / "tracked.txt").write_text("two\n")
        expect("tracked edit", repo.generate("tracked edit", tree, output), first + "-dirty")

        repo.run(tree, "commit", "-q", "-am", "second")
        second = repo.run(tree, "rev-parse", "--short=8", "HEAD")
        if second == first:
            fail("the second commit did not change HEAD")
        expect("new commit", repo.generate("new commit", tree, output), second)

        plain = base / "plain"
        plain.mkdir()
        expect("not a repository", repo.generate("not a repository", plain, output), "unknown")

        nested = tree / "nested"
        nested.mkdir()
        expect("directory inside another repository", repo.generate("directory inside another repository", nested, output), "unknown")

        inherited = {"GIT_DIR": str(tree / ".git"), "GIT_WORK_TREE": str(tree)}
        expect("inherited GIT_DIR", repo.generate("inherited GIT_DIR", plain, output, env=inherited), "unknown")

        expect("no git executable", repo.generate("no git executable", tree, output, git=""), "unknown")
    print("startup_build_rev_check: 10 cases ok")


def dll(dll_path, source_path):
    rev = read_rev(source_path)
    try:
        data = Path(dll_path).read_bytes()
    except OSError as error:
        fail(f"cannot read {dll_path}: {error}")
    wanted = {
        "narrow revision": rev.encode("ascii") + b"\0",
        "UTF-16 revision": rev.encode("utf-16-le") + b"\0\0",
        "device-create log format": LOG_FORMAT.encode("ascii") + b"\0",
    }
    for name, text in TITLE_FORMATS.items():
        wanted[name] = text.encode("utf-16-le") + b"\0\0"
    missing = [name for name, needle in wanted.items() if needle not in data]
    if missing:
        fail(f"{dll_path} (revision {rev}) lacks: {', '.join(missing)}")
    print(f"ok {dll_path}: revision {rev}, {len(wanted)} strings")


def main(argv):
    if not argv:
        controls()
    elif argv[0] == "dll" and len(argv) == 3:
        dll(argv[1], argv[2])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
