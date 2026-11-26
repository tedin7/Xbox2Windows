#!/usr/bin/env python3
"""Check tools/startup/runtime_sources.py: tester_run.ps1 uses a build from another commit only when the runtime sources
(src, cmake, CMakeLists.txt, CMakePresets.json) are the same there and at HEAD, with no local change in them.

Pure Python plus git, in a throw-away repository. Needs git; without it the check prints a notice and passes.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / "tools" / "startup" / "runtime_sources.py"


def git(tree, *args):
    return subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True, check=True).stdout.strip()


def commit(tree, files, message):
    for name, text in files.items():
        path = tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git(tree, "add", *files)
    git(tree, "commit", "-q", "-m", message)
    return git(tree, "rev-parse", "--short=8", "HEAD")


def check(tree, built):
    done = subprocess.run([sys.executable, "-B", str(TOOL), str(tree), built], capture_output=True, text=True)
    return done.returncode, done.stdout.strip()


def expect(tree, built, code, *words):
    got, text = check(tree, built)
    assert got == code and all(word in text for word in words), (built, code, words, got, text)


def main():
    if not shutil.which("git"):
        print("startup_runtime_sources_check: git not found, skipped")
        return 0
    with tempfile.TemporaryDirectory() as root:
        tree = Path(root)
        git(tree, "init", "-q")
        git(tree, "config", "user.email", "check@example.invalid")
        git(tree, "config", "user.name", "check")
        built = commit(tree, {"src/device.cpp": "int a;\n", "cmake/rev.cmake": "# rev\n", "CMakeLists.txt": "# top\n",
                              "CMakePresets.json": "{}\n", "tools/startup/run.ps1": "# run\n", "docs/a.md": "a\n"}, "first")
        expect(tree, built, 0, "same runtime sources", built)  # the build of HEAD itself

        head = commit(tree, {"tools/startup/run.ps1": "# run 2\n", "docs/a.md": "b\n", "tests/t.py": "# t\n"}, "tools only")
        expect(tree, built, 0, "the build %s has the same runtime sources as %s" % (built, head))

        (tree / "tools" / "startup" / "run.ps1").write_text("# local tools edit\n", encoding="utf-8")
        expect(tree, built, 0, "same runtime sources")
        (tree / "src" / "device.cpp").write_text("int b;\n", encoding="utf-8")
        expect(tree, built, 1, "local changes in the runtime sources", "src/device.cpp")
        git(tree, "checkout", "-q", "--", "src", "tools")

        for name in ("src/device.cpp", "cmake/rev.cmake", "CMakeLists.txt", "CMakePresets.json"):
            changed = commit(tree, {name: "changed in %s\n" % name}, "runtime change")
            expect(tree, built, 1, "the runtime sources differ between the build %s and %s" % (built, changed), name)
        expect(tree, changed, 0, "same runtime sources")  # a build of the newest commit
        added = commit(tree, {"src/new_stub.cpp": "int c;\n"}, "new runtime file")
        expect(tree, changed, 1, "differ", "src/new_stub.cpp")
        expect(tree, added, 0, "same runtime sources")

        expect(tree, built + "-dirty", 1, "made from a checkout with local changes")
        expect(tree, "unknown", 1, "no source revision")
        expect(tree, "0000000f", 1, "is not in this checkout")
        expect(tree, "not-a-rev", 1, "is not a commit")
        expect(tree / "missing", built, 2, "cannot compare")
    print("startup_runtime_sources_check: tools-only commits and local tools edits reuse the build; a change or local "
          "edit in src, cmake, CMakeLists.txt or CMakePresets.json, a dirty or unknown build revision and a commit not "
          "in the checkout are refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
