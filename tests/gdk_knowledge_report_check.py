"""Synthetic controls for the GDK knowledge merge and coverage report."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "gdk_knowledge_report.py"
FACT_FIELDS = {"module", "kind", "name", "value", "source", "confidence", "targets"}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_report_tree(root):
    write_json(
        root / "layouts" / "alpha.json",
        {
            "module": "alpha",
            "classes": [
                {"name": "Shared", "size": 16, "members": [{"offset": 0, "text": "x"}]},
                {"name": "OnlyA", "size": 8, "members": []},
                {"name": "Cross", "size": 8, "members": []},
            ],
        },
    )
    write_json(
        root / "layouts" / "beta.json",
        {"module": "beta", "classes": [{"name": "Cross", "size": 16, "members": []}]},
    )
    write_json(
        root / "abi" / "alpha.json",
        {
            "module": "alpha",
            "facts": [
                {"kind": "sizeof", "name": "Shared", "value": 32},
                {"kind": "struct", "name": "AbiOnly", "value": {"size": 4}},
                {"kind": "enum", "name": "Color", "value": 1},
                {"kind": "constant", "name": "Magic", "value": 1},
            ],
        },
    )
    write_json(
        root / "abi" / "alpha-more.json",
        {
            "module": "alpha",
            "facts": [
                {"kind": "enum", "name": "Color", "value": 2},
                {"kind": "constant", "name": "Magic", "value": 2},
            ],
        },
    )
    write_json(
        root / "exports" / "alpha-a.json",
        {
            "module": "alpha",
            "exports": [
                {"name": "Foo", "ordinal": 1, "rva": "0x10"},
                {"name": "Bar", "ordinal": 3, "rva": "0x20"},
            ],
        },
    )
    write_json(
        root / "exports" / "alpha-b.json",
        {
            "module": "alpha",
            "exports": [
                {"name": "Foo", "ordinal": 2, "rva": "0x30"},
                {"name": "Bar", "ordinal": 3, "rva": "0x20"},
            ],
        },
    )
    write_json(
        root / "behavior" / "alpha.json",
        {
            "module": "alpha",
            "observations": [
                {"kind": "value", "name": "Bar", "value": 42},
                {"kind": "behavior", "name": "OnlyA", "value": "ok"},
            ],
        },
    )
    write_json(
        root / "platform" / "windows.json",
        {
            "platform": "windows-x64",
            "defines": ["ALPHA=1"],
            "includes": ["alpha.h"],
            "libs": ["alpha.lib"],
            "flags": ["/W4"],
        },
    )
    write_json(
        root / "shaders" / "corpus.json",
        {
            "module": "alpha",
            "cases": [
                {"name": "ps-case", "stage": "pixel"},
                {"name": "cs-case", "stage": "compute"},
            ],
        },
    )
    write_text(
        root / "defs" / "alpha.def",
        "LIBRARY alpha\n"
        "EXPORTS\n"
        "    Bar @3\n"
        "    MissingThing @4\n"
        "    XMemAllocDefault=Bar @5\n"
        "    x2w_internal_helper @6\n",
    )
    write_json(
        root / "defs-map.json",
        {"alpha": ["Bar", "MappingOnly", "x2w_mapping_internal"]},
    )


def run_report(root, output, *, defs=None, require=None, check=False, markdown=None):
    json_path = Path(output) / "coverage.json"
    markdown_path = Path(markdown) if markdown else Path(output) / "report.md"
    command = [
        sys.executable,
        "-B",
        str(TOOL),
        "report",
        "--root",
        str(root),
        "--json",
        str(json_path),
        "--markdown",
        str(markdown_path),
    ]
    if defs is not None:
        command.extend(["--defs", str(defs)])
    if require is not None:
        command.extend(["--require", require])
    if check:
        command.append("--check")
    process = subprocess.run(command, capture_output=True, text=True)
    return process, json_path, markdown_path


def check_report():
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "knowledge"
        make_report_tree(root)

        process, json_path, markdown_path = run_report(
            root, base / "out", defs=root / "defs"
        )
        assert process.returncode == 0, process.stderr
        report = json.loads(json_path.read_text(encoding="utf-8"))
        markdown = markdown_path.read_text(encoding="utf-8")

        assert report["schema"] == "xbox2windows.gdk_knowledge/1"
        assert report["generated_by"] == {
            "tool": "gdk_knowledge_report",
            "version": 1,
        }
        assert set(report["categories"]) == {
            "layouts",
            "abi",
            "exports",
            "behavior",
            "platform",
            "shaders",
        }
        assert all(info["present"] for info in report["categories"].values())
        assert report["missing"] == []
        assert report["required"] == []
        assert report["required_missing"] == []

        summary = report["summary"]
        assert summary["modules"] == 3
        assert summary["types"] == 5
        assert summary["classes_types"] == 5
        assert summary["exports"] == 2
        assert summary["export_rows"] == 4
        assert summary["runtime_facts"] == 2
        assert summary["shader_facts"] == 2
        assert summary["platforms"] == 1
        assert summary["facts"] == 19
        assert summary["conflicts"] == 5
        assert summary["shims"] == 3
        assert summary["shims_without_fact"] == 1
        assert report["categories"]["layouts"]["facts"] == 4
        assert report["categories"]["abi"]["facts"] == 6
        assert report["categories"]["exports"]["facts"] == 4

        assert len(report["facts"]) == 19
        for fact in report["facts"]:
            assert set(fact) == FACT_FIELDS
            assert fact["module"]
            assert fact["kind"]
            assert fact["name"]
            assert isinstance(fact["confidence"], str)
            assert isinstance(fact["targets"], list)
            assert not str(fact["source"]).startswith(str(base))
        assert report["facts"] == sorted(
            report["facts"],
            key=lambda fact: (
                fact["module"].casefold(),
                fact["module"],
                fact["kind"].casefold(),
                fact["kind"],
                fact["name"].casefold(),
                fact["name"],
                json.dumps(
                    fact["value"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
                ),
                json.dumps(
                    fact["source"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
                ),
            ),
        )

        shared = next(
            fact
            for fact in report["facts"]
            if fact["module"] == "alpha"
            and fact["kind"] == "layout"
            and fact["name"] == "Shared"
        )
        assert shared["value"] == {
            "size": 16,
            "members": [{"offset": 0, "text": "x"}],
        }
        assert shared["source"] == "layouts/alpha.json"
        assert shared["confidence"] == "header"

        behavior = next(
            fact
            for fact in report["facts"]
            if fact["name"] == "Bar" and fact["kind"] == "behavior"
        )
        assert behavior["value"] == 42
        assert behavior["confidence"] == "runtime"

        kinds = [conflict["kind"] for conflict in report["conflicts"]]
        assert kinds == ["value", "size", "ordinal", "enum", "type-size"], kinds
        by_kind = {conflict["kind"]: conflict for conflict in report["conflicts"]}
        assert by_kind["size"]["module"] == "alpha"
        assert by_kind["size"]["name"] == "Shared"
        assert {item["value"] for item in by_kind["size"]["values"]} == {16, 32}
        assert by_kind["ordinal"]["module"] == "alpha"
        assert by_kind["ordinal"]["name"] == "Foo"
        assert [item["value"] for item in by_kind["ordinal"]["values"]] == [1, 2]
        assert by_kind["enum"]["name"] == "Color"
        assert {item["value"] for item in by_kind["enum"]["values"]} == {1, 2}
        assert by_kind["type-size"]["name"] == "Cross"
        assert by_kind["type-size"]["modules"] == ["alpha", "beta"]
        assert {item["module"] for item in by_kind["type-size"]["values"]} == {"alpha", "beta"}
        assert by_kind["value"]["name"] == "Magic"
        for conflict in report["conflicts"]:
            assert conflict["message"]
            assert conflict["sources"] == sorted(set(conflict["sources"]))

        shims = report["shims"]
        assert shims["exports"] == 3
        assert shims["covered"] == 2
        assert shims["internal"] == 1
        assert [
            (item["module"], item["name"], item["alias"])
            for item in shims["without_fact"]
        ] == [("alpha", "MissingThing", None)]

        modules = {module["module"]: module for module in report["modules"]}
        assert set(modules) == {"alpha", "beta", "windows-x64"}
        assert modules["alpha"]["types"] == 4
        assert modules["alpha"]["exports"] == 2
        assert modules["alpha"]["runtime_facts"] == 2
        assert modules["alpha"]["shader_facts"] == 2
        assert modules["windows-x64"]["platform_facts"] == 1

        assert markdown.startswith("# GDK knowledge coverage\n")
        assert "| Modules | 3 |" in markdown
        assert "| Classes/types | 5 |" in markdown
        assert "| Runtime facts | 2 |" in markdown
        assert "| Conflicts | 5 |" in markdown
        assert "| ordinal | alpha | Foo |" in markdown
        assert "MissingThing" in markdown
        assert "A header fact does not prove the runtime." in markdown
        assert markdown.endswith("\n") and not markdown.endswith("\n\n")

        second_process, second_json, second_markdown = run_report(
            root, base / "out-again", defs=root / "defs"
        )
        assert second_process.returncode == 0, second_process.stderr
        assert json_path.read_bytes() == second_json.read_bytes()
        assert markdown_path.read_bytes() == second_markdown.read_bytes()

        mapping_process, mapping_json, _ = run_report(
            root, base / "out-mapping", defs=root / "defs-map.json"
        )
        assert mapping_process.returncode == 0, mapping_process.stderr
        mapping_report = json.loads(mapping_json.read_text(encoding="utf-8"))
        assert mapping_report["summary"]["shims"] == 2
        assert mapping_report["summary"]["shims_without_fact"] == 1
        assert [
            (item["module"], item["name"])
            for item in mapping_report["shims"]["without_fact"]
        ] == [("alpha", "MappingOnly")]
        assert mapping_report["shims"]["internal"] == 1

        checked, checked_json, checked_markdown = run_report(
            root, base / "out-check", defs=root / "defs", check=True
        )
        assert checked.returncode == 1
        assert checked_json.is_file()
        assert checked_markdown.is_file()
        assert "conflict" in checked.stderr


def check_require():
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "clean"
        write_json(
            root / "layouts" / "one.json",
            {"module": "one", "classes": [{"name": "Only", "size": 4, "members": []}]},
        )

        passed, passed_json, _ = run_report(
            root, base / "out-pass", require="layouts", check=True
        )
        assert passed.returncode == 0, passed.stderr
        passed_report = json.loads(passed_json.read_text(encoding="utf-8"))
        assert passed_report["required"] == ["layouts"]
        assert passed_report["required_missing"] == []
        assert passed_report["missing"] == ["abi", "exports", "behavior", "platform", "shaders"]

        failed, failed_json, failed_markdown = run_report(
            root, base / "out-fail", require="layouts,exports", check=True
        )
        assert failed.returncode == 1
        assert "exports" in failed.stderr
        failed_report = json.loads(failed_json.read_text(encoding="utf-8"))
        assert failed_report["required"] == ["layouts", "exports"]
        assert failed_report["required_missing"] == ["exports"]
        assert "Required categories missing: exports." in failed_markdown.read_text(
            encoding="utf-8"
        )

        warned, warned_json, warned_markdown = run_report(
            root, base / "out-warn", require="exports"
        )
        assert warned.returncode == 0, warned.stderr
        warned_report = json.loads(warned_json.read_text(encoding="utf-8"))
        assert warned_report["required_missing"] == ["exports"]
        assert "Required categories missing: exports." in warned_markdown.read_text(
            encoding="utf-8"
        )


def check_module_case():
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "case"
        write_json(
            root / "layouts" / "one.json",
            {"module": "Alpha", "classes": [{"name": "One", "size": 4, "members": []}]},
        )
        write_json(
            root / "layouts" / "two.json",
            {"module": "alpha", "classes": [{"name": "Two", "size": 8, "members": []}]},
        )
        process, json_path, _ = run_report(root, base / "out")
        assert process.returncode == 0, process.stderr
        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert [module["module"] for module in report["modules"]] == ["alpha"]
        assert report["modules"][0]["types"] == 2
        assert report["summary"]["modules"] == 1


def check_errors():
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)

        missing_root, _, _ = run_report(base / "absent", base / "out-absent")
        assert missing_root.returncode == 2
        assert "does not exist" in missing_root.stderr

        malformed = base / "malformed"
        write_text(malformed / "layouts" / "bad.json", "{")
        malformed_process, _, _ = run_report(malformed, base / "out-malformed")
        assert malformed_process.returncode == 2
        assert "invalid JSON" in malformed_process.stderr

        wrong_shape = base / "wrong-shape"
        write_json(wrong_shape / "layouts" / "bad.json", {"classes": {}})
        wrong_process, _, _ = run_report(wrong_shape, base / "out-wrong")
        assert wrong_process.returncode == 2
        assert "classes" in wrong_process.stderr

        valid = base / "valid"
        write_json(
            valid / "layouts" / "one.json",
            {"module": "one", "classes": [{"name": "Only", "size": 4, "members": []}]},
        )
        unknown_require, _, _ = run_report(
            valid, base / "out-unknown", require="layouts,unknown"
        )
        assert unknown_require.returncode == 2
        assert "unknown category" in unknown_require.stderr

        absent_defs, _, _ = run_report(
            valid, base / "out-defs", defs=base / "absent-defs"
        )
        assert absent_defs.returncode == 2
        assert "does not exist" in absent_defs.stderr

        bad_mapping = base / "bad-mapping.json"
        write_text(bad_mapping, "[")
        bad_mapping_process, _, _ = run_report(
            valid, base / "out-map", defs=bad_mapping
        )
        assert bad_mapping_process.returncode == 2
        assert "invalid JSON" in bad_mapping_process.stderr

        same_path = base / "same.md"
        same_process, _, _ = run_report(
            valid, base / "out-same", markdown=same_path
        )
        assert same_process.returncode == 0, same_process.stderr

        json_path = base / "collision.json"
        collision = subprocess.run(
            [
                sys.executable,
                "-B",
                str(TOOL),
                "report",
                "--root",
                str(valid),
                "--json",
                str(json_path),
                "--markdown",
                str(json_path),
            ],
            capture_output=True,
            text=True,
        )
        assert collision.returncode == 2
        assert "different paths" in collision.stderr


def main():
    check_report()
    check_require()
    check_module_case()
    check_errors()
    print("gdk_knowledge_report_check: ok")


if __name__ == "__main__":
    main()
