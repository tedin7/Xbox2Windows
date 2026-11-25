"""Compare the C++ decode_lanes paths with the vendored Python decoder (tools/_bcpack_rans).

The lane check executable writes a sample of streams with each compared path's result: the
repository vectors, the mode-6 vector and synthetic streams from tests/bcpack_encoder.h, each whole,
truncated and with bits flipped. The C++ paths and the C++ encoder share the decoder's lane code; the
Python decoder is a separate implementation, so it anchors the accept/reject decision and the symbols.

usage: bcpack_lanes_python_check.py LANES_CHECK_EXE VECTOR_DIR [--path NAME]
Exit 77 when the executable reports that this CPU lacks the path.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from _bcpack_rans import BCPackError, decode_lanes  # noqa: E402


def python_decode(data, ab, target):
    """The vendored reference with the C++ decoder's trailing-byte rule; None when it rejects."""
    try:
        result = decode_lanes(data, list(ab), [target] * len(ab))
    except BCPackError:
        return None
    if any(data[result["consumed"]:]):
        return None
    return [value for lane in result["syms"] for value in lane]


def main():
    if len(sys.argv) not in (3, 5) or (len(sys.argv) == 5 and sys.argv[3] != "--path"):
        print(__doc__, file=sys.stderr)
        return 64
    with tempfile.TemporaryDirectory() as directory:
        sample = Path(directory) / "sample.jsonl"
        run = subprocess.run([sys.argv[1], sys.argv[2], *sys.argv[3:], "--emit", str(sample)],
                             capture_output=True, text=True)
        if run.returncode == 77:
            print(run.stdout.strip())
            return 77
        if run.returncode != 0:
            print(run.stdout, run.stderr, file=sys.stderr)
            return 1
        lines = sample.read_text().splitlines()
    mismatches = accepted = 0
    for line in lines:
        case = json.loads(line)
        data = bytes.fromhex(case["hex"])
        expected = python_decode(data, case["ab"], case["target"])
        accepted += expected is not None
        for path, result in case["paths"].items():
            got = result["symbols"] if result["ok"] else None
            if got != expected:
                mismatches += 1
                if mismatches <= 10:
                    print(f"{path} path, {len(data)} bytes, widths {case['ab']}, {case['target']} symbols: "
                          f"Python {'accepts' if expected is not None else 'rejects'}, "
                          f"C++ {'accepts' if result['ok'] else 'rejects'}"
                          f"{' with other symbols' if expected is not None and result['ok'] else ''}",
                          file=sys.stderr)
    paths = sorted(json.loads(lines[0])["paths"]) if lines else []
    print(f"{len(lines)} streams ({accepted} accepted by the Python decoder), paths {', '.join(paths)}: "
          f"{mismatches} mismatches")
    return 0 if lines and mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
