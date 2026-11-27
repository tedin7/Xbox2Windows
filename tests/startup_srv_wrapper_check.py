"""Only the SRV wrapper may reference the native D3D12 creation method."""

import re
from pathlib import Path

WRAPPER = Path("startup_stubs/create_srv.h")
# Skip comments and literals so diagnostics and documentation can name the API.
TOKENS = re.compile(
    r'R"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)"'
    r'|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    r'|(?P<method>\bCreateShaderResourceView\b)',
    re.DOTALL,
)


def check_source_tree(source):
    errors = []
    wrapper_references = 0
    for path in sorted(source.rglob("*")):
        if path.suffix.lower() not in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl", ".ipp"}:
            continue
        relative = path.relative_to(source)
        text = path.read_text(encoding="utf-8-sig")
        for token in TOKENS.finditer(text):
            if token.lastgroup != "method":
                continue
            if relative == WRAPPER:
                wrapper_references += 1
            else:
                line = text.count("\n", 0, token.start()) + 1
                errors.append(f"src/{relative.as_posix()}:{line}: use x2w_create_srv")
    if wrapper_references != 1:
        errors.append(f"src/{WRAPPER.as_posix()}: expected one native SRV method reference, found {wrapper_references}")
    return errors


def main():
    errors = check_source_tree(Path(__file__).resolve().parents[1] / "src")
    if errors:
        print("\n".join(errors))
        return 1
    print("PASS: native CreateShaderResourceView appears only in its wrapper")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
