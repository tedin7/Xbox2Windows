#!/usr/bin/env python3
"""Parse MSVC /d1reportAllClassLayout logs and compare two layout sets.

The logs are produced outside the repository from GDK headers. This tool keeps
only sizes, offsets and member names, so the JSON can live in the local
knowledge base without copying header text.

Commands:
  parse <log>... --json <out> [--module M] [--platform P] [--header H]
        [--header-sha256 SHA] [--msvc VERSION] [--prefix P]...
  compare <left.json> <right.json> [--json <out>]

compare exits 1 when a class size or a named member offset differs.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


HEADER_RE = re.compile(r"^(?:class|struct|union)\s+(.+?)\s+size\((\d+)\):")
MEMBER_RE = re.compile(r"^\s*(\d+)\s*\|\s*(.*?)\s*$")
ALIGNMENT_RE = re.compile(r"^\s*\|\s*<alignment member> \(size=(\d+)\)")
BASE_RE = re.compile(r"\+---\s*\(base class\s+(.+?)\)")
BITS_RE = re.compile(r":\s*(\d+)\s*$")
ARRAY_RE = re.compile(r"^(.*?)\s+([A-Za-z_]\w*)\s*\[(\d+)\]$")
NAMED_RE = re.compile(r"^(?P<type>.+?)\s+(?P<name>[A-Za-z_]\w*)$")
VFTABLE = "{vfptr}"


def sha256(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest().upper()


def parse_member(offset, text):
    if BASE_RE.search(text):
        return {
            "offset": offset,
            "type": "",
            "name": BASE_RE.search(text).group(1).strip(),
            "bits": None,
            "kind": "base",
        }
    if VFTABLE in text:
        return {"offset": offset, "type": "void*", "name": VFTABLE, "bits": None, "kind": "vftable"}

    rest = text
    bits = None
    match = BITS_RE.search(rest)
    if match:
        bits = int(match.group(1))
        rest = rest[: match.start()].rstrip()

    array = None
    match = ARRAY_RE.match(rest)
    if match:
        rest, count = match.group(1).strip(), int(match.group(3))
        array = count
        name = match.group(2)
        return {
            "offset": offset,
            "type": rest,
            "name": name,
            "bits": bits,
            "kind": "field",
            "array": array,
        }

    match = NAMED_RE.match(rest)
    if match:
        return {
            "offset": offset,
            "type": match.group("type").strip(),
            "name": match.group("name"),
            "bits": bits,
            "kind": "field",
        }
    if re.fullmatch(r"[A-Za-z_]\w*", rest):
        return {
            "offset": offset,
            "type": "unknown",
            "name": rest,
            "bits": bits,
            "kind": "field",
        }
    return {
        "offset": offset,
        "type": rest.strip(),
        "name": "",
        "bits": bits,
        "kind": "unnamed",
    }


def parse_log(path):
    classes = []
    current = None
    alignment = 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        header = HEADER_RE.match(line)
        if header:
            if current is not None:
                current["alignment"] = alignment
            name = header.group(1).strip()
            current = {
                "name": name,
                "size": int(header.group(2)),
                "bases": [],
                "members": [],
                "alignment": 0,
                "nested_in": name.rsplit("::", 1)[0] if "::" in name else None,
            }
            alignment = 0
            classes.append(current)
            continue
        if current is None:
            continue
        if line.strip().startswith("+---"):
            continue
        align = ALIGNMENT_RE.match(line)
        if align:
            alignment = int(align.group(1))
            current["members"].append(
                {
                    "offset": current["size"] - alignment,
                    "type": "",
                    "name": "<alignment>",
                    "bits": None,
                    "kind": "alignment",
                }
            )
            continue
        member = MEMBER_RE.match(line)
        if not member:
            continue
        text = re.sub(r"^(?:\|\s*)+", "", member.group(2))
        if not text.strip():
            continue
        parsed = parse_member(int(member.group(1)), text)
        if parsed["kind"] == "base":
            current["bases"].append({"offset": parsed["offset"], "name": parsed["name"]})
        else:
            current["members"].append(parsed)
    if current is not None:
        current["alignment"] = alignment
    for block in classes:
        seen = set()
        unique = []
        for member in block["members"]:
            key = (member["offset"], member["name"], member["kind"])
            if key not in seen:
                seen.add(key)
                unique.append(member)
        block["members"] = unique
    return classes


def parse_command(args):
    classes = []
    sources = []
    for path in args.logs:
        block = parse_log(path)
        classes.extend(block)
        sources.append({"log": str(path), "sha256": sha256(path), "classes": len(block)})
    if args.prefix:
        classes = [item for item in classes if any(item["name"].startswith(prefix) for prefix in args.prefix)]
    document = {
        "sdk": args.sdk,
        "module": args.module,
        "platform": args.platform,
        "header": args.header,
        "header_sha256": args.header_sha256 or (sha256(args.header) if args.header else ""),
        "msvc": args.msvc,
        "source": "msvc-layout",
        "logs": sources,
        "classes": classes,
        "stats": {"classes": len(classes), "bytes": sum(item["size"] for item in classes)},
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document["stats"]))
    return 0


def index_classes(document):
    return {item["name"]: item for item in document.get("classes", [])}


def compare_command(args):
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    left_classes = index_classes(left)
    right_classes = index_classes(right)
    changes = []
    for name in sorted(set(left_classes) | set(right_classes)):
        old = left_classes.get(name)
        new = right_classes.get(name)
        if old is None:
            changes.append({"class": name, "kind": "added", "right_size": new["size"]})
            continue
        if new is None:
            changes.append({"class": name, "kind": "removed", "left_size": old["size"]})
            continue
        if old["size"] != new["size"]:
            changes.append(
                {"class": name, "kind": "size", "left_size": old["size"], "right_size": new["size"]}
            )
        old_members = {item["name"]: item["offset"] for item in old["members"] if item.get("name") and item["kind"] == "field"}
        new_members = {item["name"]: item["offset"] for item in new["members"] if item.get("name") and item["kind"] == "field"}
        for member in sorted(set(old_members) | set(new_members)):
            if old_members.get(member) != new_members.get(member):
                changes.append(
                    {
                        "class": name,
                        "member": member,
                        "kind": "offset",
                        "left_offset": old_members.get(member),
                        "right_offset": new_members.get(member),
                    }
                )
    if args.json:
        Path(args.json).write_text(json.dumps(changes, indent=2) + "\n", encoding="utf-8")
    if not changes:
        print("no layout differences")
        return 0
    for change in changes:
        print(json.dumps(change, sort_keys=True))
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    parse = sub.add_parser("parse", help="parse MSVC layout logs")
    parse.add_argument("logs", nargs="+")
    parse.add_argument("--json", required=True)
    parse.add_argument("--sdk", default="261000")
    parse.add_argument("--module", default="")
    parse.add_argument("--platform", default="")
    parse.add_argument("--header", default="")
    parse.add_argument("--header-sha256", default="")
    parse.add_argument("--msvc", default="")
    parse.add_argument("--prefix", action="append", default=[])
    parse.set_defaults(func=parse_command)

    compare = sub.add_parser("compare", help="compare two layout JSON files")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument("--json")
    compare.set_defaults(func=compare_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
