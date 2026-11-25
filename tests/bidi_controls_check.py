"""Reject explicit bidirectional text controls in tracked source, scripts and build/CI configuration.

The explicit controls are the embeddings and overrides U+202A..U+202E, the isolates
U+2066..U+2069 and the Arabic letter mark U+061C. They reorder how text is displayed without
changing what a compiler or an interpreter reads, so a reviewer can see different code from
what runs. Each finding is reported as path:line: code point and name. Files are read as UTF-8
with replacement, so the check does not enforce an encoding, and Markdown documentation is out
of scope.
"""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import unicodedata

CONTROLS = frozenset([0x061C, *range(0x202A, 0x202F), *range(0x2066, 0x206A)])
SOURCE_SUFFIXES = frozenset({
    ".asm", ".c", ".cc", ".cpp", ".cxx", ".def", ".h", ".hh", ".hlsl", ".hlsli", ".hpp", ".inc",
    ".inl", ".py", ".s", ".spvasm", ".zig",            # sources
    ".bat", ".cmd", ".ps1", ".psm1", ".sh",            # scripts
    ".cmake",                                          # build scripts
})
CONFIG_NAMES = frozenset({"CMakeLists.txt", "CMakePresets.json", "requirements-dev.txt", ".gitattributes"})


def in_scope(path):
    """True for source files, scripts, and build or CI configuration."""
    path = PurePosixPath(path)
    return (path.suffix.lower() in SOURCE_SUFFIXES or path.name in CONFIG_NAMES
            or (path.parts[:1] == (".github",) and path.suffix.lower() in {".yml", ".yaml"}))


def findings(root, paths):
    """Yield one 'path:line: U+XXXX NAME' line per control character."""
    for path in sorted(paths):
        text = (Path(root) / path).read_bytes().decode("utf-8", errors="replace")
        # Count lines on LF only, as editors and git do; splitlines() also breaks on CR, FF and U+2028.
        for number, line in enumerate(text.split("\n"), start=1):
            for char in line:
                if ord(char) in CONTROLS:
                    yield f"{path}:{number}: U+{ord(char):04X} {unicodedata.name(char)}"


def tracked_files(root):
    # An inherited GIT_DIR or GIT_WORK_TREE, as in a Git hook, would point at another repository.
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("GIT_DIR", "GIT_WORK_TREE")}
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True,
                            env=environment)
    return [path for path in result.stdout.decode("utf-8").split("\0") if path]


def self_test():
    override = chr(0x202E)
    isolate = chr(0x2066)
    arabic_letter_mark = chr(0x061C)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        files = {
            "tools/hidden.py": f"value = 1\nif user != 'admin{override} ':  # comment\n",
            # CRLF line ends and a form feed must not shift the reported line.
            "tests/isolate.cpp": f"int x = 0;\r\n\x0cint z = 0;{isolate}\r\n",
            "tests/ordinary.cpp": "// 64×64 texels, café, naïve – ≡\nint y = 0;\n",
            "docs/notes.md": f"Markdown is out of scope {override}\n",
            "docs/coverage/table.tsv": f"a data table is out of scope\t{override}\n",
            "CMakeLists.txt": "project(sample)\n",
            ".github/workflows/ci.yml": f"name: ci{override}\n",
            "tools/mark.py": f"label = 'r\u00e9sum\u00e9{arabic_letter_mark}'\n",
            # Zig, SPIR-V assembly and assembler sources are source files like the ones above.
            "tests/cpu/archive/reader.zig": f"const a = 1;{override}\n",
            "tests/spirv_window/case.spvasm": f"; note\n%1 = OpTypeVoid {isolate}\n",
            "tests/kernel.s": f"s_mov_b32 s0, 0{arabic_letter_mark}\n",
        }
        for name, text in files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(text.encode("utf-8"))  # exact bytes, no newline translation
        scoped = [name for name in files if in_scope(name)]
        assert "docs/notes.md" not in scoped and "docs/coverage/table.tsv" not in scoped
        assert list(findings(root, scoped)) == [
            ".github/workflows/ci.yml:1: U+202E RIGHT-TO-LEFT OVERRIDE",
            "tests/cpu/archive/reader.zig:1: U+202E RIGHT-TO-LEFT OVERRIDE",
            "tests/isolate.cpp:2: U+2066 LEFT-TO-RIGHT ISOLATE",
            "tests/kernel.s:1: U+061C ARABIC LETTER MARK",
            "tests/spirv_window/case.spvasm:2: U+2066 LEFT-TO-RIGHT ISOLATE",
            "tools/hidden.py:2: U+202E RIGHT-TO-LEFT OVERRIDE",
            "tools/mark.py:1: U+061C ARABIC LETTER MARK",
        ], list(findings(root, scoped))
    # Two ranges and one code point: the ten explicit controls, nothing more.
    assert sorted(CONTROLS) == [0x061C, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
                                0x2066, 0x2067, 0x2068, 0x2069]
    print("Self-test passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return 0
    root = Path(__file__).resolve().parents[1]
    paths = [path for path in tracked_files(root) if in_scope(path)]
    found = list(findings(root, paths))
    if found:
        print("\n".join(found))
        return 1
    print(f"Checked {len(paths)} tracked source, script and build/CI files: no bidirectional controls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
