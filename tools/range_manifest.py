"""Write a range manifest for cpu_switch_aot --bind-ranges from a linker map's source labels.

For each NAME, the function's start is the public symbol NAME and its end the public symbol
NAME_end, both read from the MSVC /MAP file of IMAGE. The manifest is bound to IMAGE by its SHA-256,
so it only applies to that exact build.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

SYMBOL = re.compile(r"^\s*[0-9a-fA-F]{4}:[0-9a-fA-F]{8}\s+(\S+)\s+([0-9a-fA-F]{16})\b")
BASE = re.compile(r"Preferred load address is ([0-9a-fA-F]{16})")


def map_symbols(text):
    base = BASE.search(text)
    if not base:
        raise ValueError("linker map without a preferred load address")
    base = int(base.group(1), 16)
    symbols = {}
    for line in text.splitlines():
        match = SYMBOL.match(line)
        if match:
            symbols.setdefault(match.group(1), int(match.group(2), 16) - base)
    return symbols


def manifest(image, map_file, names):
    if not names:
        raise ValueError("at least one function name is required")
    seen = set()
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("function name must not be empty")
        if name in seen:
            raise ValueError(f"duplicate function name: {name}")
        seen.add(name)
    symbols = map_symbols(Path(map_file).read_text(encoding="utf-8", errors="replace"))
    functions = []
    for name in names:
        if name not in symbols or f"{name}_end" not in symbols:
            raise ValueError(f"linker map has no {name} or {name}_end")
        start = symbols[name]
        end = symbols[f"{name}_end"]
        if end <= start:
            raise ValueError(f"invalid interval for {name}: end ({end:#x}) <= start ({start:#x})")
        functions.append({"name": name, "start": f"{start:#x}", "end": f"{end:#x}"})
    return {"image_sha256": hashlib.sha256(Path(image).read_bytes()).hexdigest(), "functions": functions}


def _write_atomic_text(output_path, text):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_path, output)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("map", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args(argv)
    try:
        data = manifest(args.image, args.map, args.names)
        payload = json.dumps(data, indent=2) + "\n"
        _write_atomic_text(args.output, payload)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
