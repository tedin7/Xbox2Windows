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

    [[patch]]
    id = "s001"
    kind = "shader"              # the translation files of one shader cache key
    why = "<issue or root cause>"
    remove_when = "<condition>"
    key = "0123456789abcdef"     # the cache key, 16 lowercase hex digits
    pipeline = "compute"         # graphics, compute or mesh86
    revision = "x2w-startup-shader-29cs"
    files = [{ ext = ".txt", file = "s001.txt", sha256 = "<sha256>" },
             { ext = ".cs.cso", file = "s001.cs.cso", sha256 = "<sha256>" }]

The whole manifest is validated and the input's SHA-256 compared before anything is returned, so
a manifest that fails any check applies nothing. The memory and shader kinds are defined, checked
for shape only; nothing applies them yet.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tomllib

SHA256 = re.compile(r"[0-9a-f]{64}")
HEX = re.compile(r"(?:[0-9a-f]{2})+")
KEY = re.compile(r"[0-9a-f]{16}")
# Names Windows opens as a device instead of a file in the directory, alone or before an extension (Microsoft,
# "Naming Files, Paths, and Namespaces"). Matched against the part before the first dot, trailing spaces dropped.
DEVICE = re.compile(r"(?:con|prn|aux|nul|com[0-9\u00b9\u00b2\u00b3]|lpt[0-9\u00b9\u00b2\u00b3]|conin\$|conout\$)",
                    re.IGNORECASE)
COMMON = {"id", "kind", "why", "remove_when"}
# Every key of a kind is required. hook and data get their tables with their stages.
KINDS = {"memory": {"rva", "expect_sha256", "bytes"}, "shader": {"key", "pipeline", "revision", "files"}}
# Shader patches, per pipeline: the form of the runtime's key revision (src/startup_stubs/d3d12_draw.inl and
# mesh86_cache.inl; <n> is the number each revision bump raises), the translation files the runtime reads for one
# key, and the ones without which it has no translation. A later consumer must check which is current. compute
# also needs .cs.hlsl when its .txt has no native line, and mesh86 reads .ps.cso only for a request with a pixel
# shader; this loader reads neither, so both stay optional.
PIPELINES = {
    "graphics": ("x2w-startup-shader-<n>", (".txt", ".vs.cso", ".ps.cso"), (".txt", ".vs.cso", ".ps.cso")),
    "compute": ("x2w-startup-shader-<n>cs", (".txt", ".cs.cso", ".cs.hlsl"), (".txt", ".cs.cso")),
    "mesh86": ("x2w-startup-mesh-86-<n>", (".mesh86.txt", ".ms.cso", ".ps.cso"), (".mesh86.txt", ".ms.cso")),
}
REVISIONS = {name: re.compile(re.escape(form).replace("<n>", "[1-9][0-9]*"))
             for name, (form, _, _) in PIPELINES.items()}


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


def _file_name(value, where):
    """A bare file name, for a later consumer to resolve in the manifest's directory.

    That consumer is the runtime, which runs on Windows, so a name that Windows would not resolve to a file of that
    name in the directory is refused too: one it cannot hold, one whose trailing dot or space it drops, a device.
    """
    name = _text(value, where)
    if any(c in name for c in "/\\:") or name in (".", ".."):
        raise ValueError(f"{where} {name!r} must be a bare file name, without a directory, drive or '..'")
    if any(c in '<>"|?*' for c in name) or any(ord(c) < 32 for c in name):
        raise ValueError(f"{where} {name!r} has a character a Windows file name cannot hold")
    if name.endswith((".", " ")):
        raise ValueError(f"{where} {name!r} ends in a dot or a space, which Windows drops")
    if DEVICE.fullmatch(name.split(".", 1)[0].rstrip(" ")):
        raise ValueError(f"{where} {name!r} is a Windows device name")
    return name


def _shader(entry, where):
    """The key, pipeline, revision and files of a shader patch, checked for shape only."""
    key = entry["key"]
    if not isinstance(key, str) or not KEY.fullmatch(key):
        raise ValueError(f"{where} key must be 16 lowercase hex digits")
    pipeline = entry["pipeline"]
    if not isinstance(pipeline, str) or pipeline not in PIPELINES:
        raise ValueError(f"{where} pipeline must be one of: {', '.join(PIPELINES)}")
    form, extensions, required = PIPELINES[pipeline]
    revision = entry["revision"]
    if not isinstance(revision, str) or not REVISIONS[pipeline].fullmatch(revision):
        raise ValueError(f"{where} revision {revision!r} is not a {pipeline} key revision ({form})")
    if not isinstance(entry["files"], list) or not entry["files"]:
        raise ValueError(f"{where} files must be a non-empty array of tables")
    files = {}
    for index, item in enumerate(entry["files"]):
        at = f"{where} files[{index}]"
        _keys(item, {"ext", "file", "sha256"}, at)
        ext = item["ext"]
        if not isinstance(ext, str) or ext not in extensions:
            raise ValueError(f"{at} ext {ext!r} is not a {pipeline} translation file ({', '.join(extensions)})")
        if ext in files:
            raise ValueError(f"{where} lists {ext} twice")
        files[ext] = {"ext": ext, "file": _file_name(item["file"], f"{at} file"),
                      "sha256": _sha256(item["sha256"], f"{at} sha256")}
    if missing := [ext for ext in required if ext not in files]:
        raise ValueError(f"{where} {pipeline} override is missing: {', '.join(missing)}")
    return {"key": key, "pipeline": pipeline, "revision": revision, "files": list(files.values())}


def parse(text):
    """Return (input_sha256, patches) from manifest text, or raise ValueError."""
    try:
        data = tomllib.loads(text)
    except RecursionError:  # tomllib recurses once per nested array or inline table
        raise ValueError("manifest is nested too deeply to parse") from None
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
        if kind == "shader":
            patch = {"id": pid, "kind": kind, "why": _text(entry["why"], f"{where} why"),
                     "remove_when": _text(entry["remove_when"], f"{where} remove_when"), **_shader(entry, where)}
            if other := next((p["id"] for p in patches if p.get("key") == patch["key"]), None):
                raise ValueError(f"patches {other} and {pid} override the same key {patch['key']}")
            patches.append(patch)
            continue
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
    return _bound(raw, target, patches, digest)


def load_data(manifest_path, data):
    """load() for an input the caller has already read: data is hashed, so the bytes that pass the gate are
    the bytes the caller goes on to use, even if the file on disk changes in between."""
    raw = Path(manifest_path).read_bytes()
    target, patches = parse(raw.decode("utf-8"))
    return _bound(raw, target, patches, hashlib.sha256(data).hexdigest())


def _bound(raw, target, patches, digest):
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
