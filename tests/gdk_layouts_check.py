#!/usr/bin/env python3
"""Checks for tools/gdk_layouts.py and tools/gdk_abi_probe.py.

The synthetic log covers the MSVC constructs the knowledge parser needs:
nested classes, a base class, a vftable, bitfields, arrays and an alignment
member. No Windows or GDK is required.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gdk_abi_probe as abi  # noqa: E402
import gdk_layouts as layouts  # noqa: E402


SYNTHETIC_LOG = """class Base\tsize(8):
\t+---
 0\t| {vfptr}
\t+---

class Outer\tsize(24):
\t+---
 0\t| +--- (base class Base)
 0\t| | {vfptr}
\t+---
 8\t| UINT Width
12\t| UINT Flags : 3
16\t| UCHAR Data[4]
  \t| <alignment member> (size=4)
\t+---

class Outer::Inner\tsize(4):
\t+---
 0\t| UINT Value
\t+---

class Single\tsize(4):
\t+---
 0\t| Value
\t+---
"""


def check(condition, what):
    if not condition:
        raise AssertionError(what)


def check_parser():
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "layout.log"
        log.write_text(SYNTHETIC_LOG, encoding="utf-8")
        classes = layouts.parse_log(log)
        by_name = {item["name"]: item for item in classes}
        check(set(by_name) == {"Base", "Outer", "Outer::Inner", "Single"}, f"classes {set(by_name)}")
        outer = by_name["Outer"]
        check(outer["size"] == 24, f"Outer size {outer['size']}")
        check(outer["bases"] == [{"offset": 0, "name": "Base"}], f"bases {outer['bases']}")
        check(outer["alignment"] == 4, f"alignment {outer['alignment']}")
        members = {item["name"]: item for item in outer["members"] if item["name"]}
        check(members["{vfptr}"]["kind"] == "vftable", "vftable kind")
        check(members["Width"]["offset"] == 8, "Width offset")
        check(members["Width"]["type"] == "UINT", "typed member")
        check(members["Flags"]["bits"] == 3, "bitfield width")
        check(members["Data"]["array"] == 4, "array count")
        check(members["<alignment>"]["kind"] == "alignment", "alignment member")
        check(by_name["Outer::Inner"]["nested_in"] == "Outer", "nested_in")
        check(by_name["Single"]["members"][0]["type"] == "unknown", "single-token member type")
        check(by_name["Single"]["members"][0]["name"] == "Value", "single-token member name")
    print("parser: base, vftable, bitfield, array and alignment passed")


def write_layouts(path, classes):
    path.write_text(
        json.dumps({"module": "synthetic", "classes": classes}, indent=2) + "\n",
        encoding="utf-8",
    )


def check_compare():
    left = [
        {"name": "A", "size": 16, "bases": [], "members": [
            {"offset": 0, "type": "UINT", "name": "X", "bits": None, "kind": "field"},
            {"offset": 8, "type": "UINT", "name": "Y", "bits": None, "kind": "field"},
        ]},
        {"name": "OnlyLeft", "size": 4, "bases": [], "members": []},
    ]
    right = [
        {"name": "A", "size": 24, "bases": [], "members": [
            {"offset": 0, "type": "UINT", "name": "X", "bits": None, "kind": "field"},
            {"offset": 12, "type": "UINT", "name": "Y", "bits": None, "kind": "field"},
            {"offset": 20, "type": "UINT", "name": "Z", "bits": None, "kind": "field"},
        ]},
        {"name": "OnlyRight", "size": 8, "bases": [], "members": []},
    ]
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        left_path = directory / "left.json"
        right_path = directory / "right.json"
        write_layouts(left_path, left)
        write_layouts(right_path, right)
        code = layouts.main(["compare", str(left_path), str(right_path), "--json", str(directory / "diff.json")])
        check(code == 1, f"compare exit {code}")
        changes = json.loads((directory / "diff.json").read_text(encoding="utf-8"))
        kinds = {(item["class"], item.get("kind")) for item in changes}
        check(("A", "size") in kinds, f"size change {kinds}")
        check(("A", "offset") in kinds, f"offset change {kinds}")
        check(("OnlyLeft", "removed") in kinds and ("OnlyRight", "added") in kinds, f"add/remove {kinds}")
        check(layouts.main(["compare", str(left_path), str(left_path)]) == 0, "identical exit")
    print("compare: size, offset, add and remove changes passed")


def check_probe():
    classes = [
        {"name": "Simple", "size": 12, "bases": [], "members": [
            {"offset": 0, "type": "UINT", "name": "A", "bits": None, "kind": "field"},
            {"offset": 4, "type": "UINT", "name": "Flags", "bits": 3, "kind": "field"},
            {"offset": 8, "type": "UINT", "name": "B", "bits": None, "kind": "field"},
        ]},
        {"name": "Derived", "size": 16, "bases": [{"offset": 0, "name": "Simple"}], "members": [
            {"offset": 0, "type": "void*", "name": "{vfptr}", "bits": None, "kind": "vftable"},
            {"offset": 12, "type": "UINT", "name": "C", "bits": None, "kind": "field"},
        ]},
    ]
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        layout_path = directory / "layout.json"
        write_layouts(layout_path, classes)
        source = directory / "probe.cpp"
        code = abi.main([
            "generate", "--layouts", str(layout_path), "--out", str(source),
            "--include", "synthetic.h", "--offsetof", "--print",
        ])
        check(code == 0, f"generate exit {code}")
        text = source.read_text(encoding="utf-8")
        check("#include <synthetic.h>" in text, "include")
        check('static_assert(sizeof(Simple) == 12, "sizeof(Simple)");' in text, "sizeof Simple")
        check('static_assert(offsetof(Simple, A) == 0, "offsetof(Simple, A)");' in text, "offsetof A")
        check("offsetof(Simple, Flags)" not in text, "bitfield offset emitted")
        check("offsetof(Derived, C)" not in text, "base-class offset emitted")
        check('std::printf("offsetof(Simple, B)=%zu\\n", offsetof(Simple, B));' in text, "print B")

        size_only = directory / "size-only.cpp"
        private_classes = classes + [
            {"name": "PrivateField", "size": 4, "bases": [], "members": [
                {"offset": 0, "type": "int", "name": "hidden", "bits": None, "kind": "field"},
            ]},
        ]
        private_layout = directory / "private.json"
        write_layouts(private_layout, private_classes)
        code = abi.main([
            "generate", "--layouts", str(private_layout), "--out", str(size_only),
            "--include", "synthetic.h", "--print",
        ])
        check(code == 0, f"size-only generate exit {code}")
        size_text = size_only.read_text(encoding="utf-8")
        check('std::printf("sizeof(PrivateField)=%zu\\n"' in size_text, "size-only sizeof")
        check("offsetof(PrivateField, hidden)" not in size_text, "print-only emitted offsetof")
        check('std::printf("offsetof(Simple, B)=%zu\\n"' not in size_text, "print-only emitted a public offsetof")

        log = directory / "probe.log"
        log.write_text("sizeof(Simple)=12\noffsetof(Simple, A)=0\n", encoding="utf-8")
        parsed = directory / "parsed.json"
        check(abi.main(["parse", str(log), "--json", str(parsed)]) == 0, "parse exit")
        facts = json.loads(parsed.read_text(encoding="utf-8"))["facts"]
        check(facts == {"sizeof(Simple)": 12, "offsetof(Simple, A)": 0}, f"facts {facts}")

        changed = directory / "changed.json"
        changed.write_text(
            json.dumps({"facts": {"sizeof(Simple)": 16, "offsetof(Simple, A)": 0}}),
            encoding="utf-8",
        )
        check(abi.main(["diff", str(parsed), str(changed)]) == 1, "diff exit")
        check(abi.main(["diff", str(parsed), str(parsed)]) == 0, "same diff exit")
    print("probe: generation, offsetof rules, parse and diff passed")


def main():
    check_parser()
    check_compare()
    check_probe()
    print("gdk_layouts_check: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
