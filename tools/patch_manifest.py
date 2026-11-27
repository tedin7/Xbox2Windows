"""Load a patch manifest and check that it is bound to the given input file.

A manifest is TOML:

    [target]
    input_sha256 = "<sha256 of the exact input file>"

    [[patch]]
    id = "p001"
    kind = "memory"
    why = "<issue or root cause>"
    remove_when = "<condition>"
    rva = 0x1010                 # first byte to replace
    expect_sha256 = "<sha256 of the len(bytes) original bytes at rva>"
    bytes = "90c3"               # the replacement, lowercase hex

The whole manifest is validated and the input's SHA-256 compared before anything is returned, so
a manifest that fails any check applies nothing. Only the memory kind is defined so far, and
nothing applies it yet.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tomllib

SHA256 = re.compile(r"[0-9a-f]{64}")
HEX = re.compile(r"(?:[0-9a-f]{2})+")
COMMON = {"id", "kind", "why", "remove_when"}
# Every key of a kind is required. hook and data get their tables with their stages.
KINDS = {"memory": {"rva", "expect_sha256", "bytes"}}


def _keys(table, keys, where):
    if not isinstance(table, dict):
        raise ValueError(f"{where} must be a table")
    if extra := sorted(set(table) - keys):
        raise ValueError(f"{where} has unknown keys: {', '.join(extra)}")
    if missing := sorted(keys - set(table)):
        raise ValueError(f"{where} is missing: {', '.join(missing)}")


def _text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _sha256(value, where):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{where} must be 64 lowercase hex digits")
    return value


def parse(text):
    """Return (input_sha256, patches) from manifest text, or raise ValueError."""
    data = tomllib.loads(text)
    _keys(data, {"target", "patch"}, "manifest")
    _keys(data["target"], {"input_sha256"}, "[target]")
    target = _sha256(data["target"]["input_sha256"], "input_sha256")
    if not isinstance(data["patch"], list) or not data["patch"]:
        raise ValueError("manifest needs at least one [[patch]]")
    patches, spans = [], []
    for index, entry in enumerate(data["patch"]):
        where = f"[[patch]] {index}"
        kind = entry.get("kind") if isinstance(entry, dict) else None
        if not isinstance(kind, str) or kind not in KINDS:  # a list or table kind is unhashable
            raise ValueError(f"{where}: kind {kind!r} is not supported")
        _keys(entry, COMMON | KINDS[kind], where)
        pid = _text(entry["id"], f"{where} id")
        if any(p["id"] == pid for p in patches):
            raise ValueError(f"duplicate patch id: {pid}")
        where = f"patch {pid}"
        rva = entry["rva"]
        if type(rva) is not int or not 0 <= rva < 1 << 32:
            raise ValueError(f"{where} rva must be an integer in [0, 2^32)")
        if not isinstance(entry["bytes"], str) or not HEX.fullmatch(entry["bytes"]):
            raise ValueError(f"{where} bytes must be a non-empty, even number of lowercase hex digits")
        replacement = bytes.fromhex(entry["bytes"])
        end = rva + len(replacement)
        if end > 1 << 32:
            raise ValueError(f"{where} span [{rva:#x}, {end:#x}) does not fit in 32-bit RVAs")
        spans.append((rva, end, pid))
        patches.append({
            "id": pid,
            "kind": kind,
            "why": _text(entry["why"], f"{where} why"),
            "remove_when": _text(entry["remove_when"], f"{where} remove_when"),
            "rva": rva,
            "expect_sha256": _sha256(entry["expect_sha256"], f"{where} expect_sha256"),
            "bytes": replacement,
        })
    spans.sort()
    for (_, end, first), (start, _, second) in zip(spans, spans[1:]):
        if start < end:
            raise ValueError(f"patches {first} and {second} overlap")
    return target, patches


def load(manifest_path, input_path):
    """Validate the manifest, then refuse it unless input_path is its target."""
    raw = Path(manifest_path).read_bytes()
    target, patches = parse(raw.decode("utf-8"))
    with open(input_path, "rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != target:
        raise ValueError(f"manifest is for another input: manifest input_sha256 {target}, input sha256 {digest}")
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "input_sha256": digest, "patches": patches}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        plan = load(args.manifest, args.input)
    except (ValueError, OSError) as error:  # TOMLDecodeError and UnicodeDecodeError are ValueErrors
        parser.exit(1, f"patch manifest: {error}\n")
    print(json.dumps({"manifest_sha256": plan["manifest_sha256"], "input_sha256": plan["input_sha256"],
                      "patch_ids": [p["id"] for p in plan["patches"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
