#!/usr/bin/env python3
"""Checks for tools/gdk_shader_corpus.py without the GDK.

The real compilation is covered by the corpus build in docs/experiments/
gdk-shader-corpus.md. This check covers case loading, hashing, manifest
verification, triple checks and the missing-dxc skip.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gdk_shader_corpus as corpus  # noqa: E402


def check(condition, what):
    if not condition:
        raise AssertionError(what)


def write_corpus(root, duplicate=False):
    (root / "case.hlsl").write_text("float4 main() : SV_Target { return 0; }\n", encoding="utf-8")
    cases = [{"name": "case", "source": "case.hlsl", "profile": "ps_6_0", "entry": "main"}]
    if duplicate:
        cases.append(dict(cases[0]))
    (root / "cases.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")


def check_loading():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_corpus(root)
        cases = corpus.load_cases(root)
        check(cases[0]["name"] == "case", f"cases {cases}")
        write_corpus(root, duplicate=True)
        try:
            corpus.load_cases(root)
        except ValueError as error:
            check("duplicate" in str(error), str(error))
        else:
            raise AssertionError("duplicate case accepted")
        (root / "case.hlsl").unlink()
        try:
            corpus.load_cases(root)
        except ValueError as error:
            check("missing" in str(error), str(error))
        else:
            raise AssertionError("missing source accepted")
    print("loading: case, duplicate and missing source passed")


def check_build_skip():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_corpus(root)
        out = root / "out"
        code = corpus.main([
            "build", "--gdk-root", str(root), "--corpus", str(root), "--out", str(out),
            "--dxc", str(root / "missing-dxc.exe"),
        ])
        check(code == corpus.SKIP, f"missing dxc exit {code}")
        code = corpus.main([
            "build", "--gdk-root", str(root), "--corpus", str(root), "--out", str(out),
            "--dxc", str(root / "missing-dxc.exe"), "--require",
        ])
        check(code == 2, f"required dxc exit {code}")
    print("build: missing dxc reports 77 and --require reports 2")


def check_verify():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "xbox.cso").write_bytes(b"xbox")
        (root / "desktop.cso").write_bytes(b"desktop")
        manifest = {
            "sdk": "261000",
            "cases": [
                {
                    "name": "case",
                    "source": "case.hlsl",
                    "source_sha256": "0" * 64,
                    "profile": "ps_6_0",
                    "entry": "main",
                    "args": [],
                    "xbox": {
                        "file": "xbox.cso",
                        "bytes": 4,
                        "sha256": corpus.sha256(root / "xbox.cso"),
                    },
                    "desktop": {
                        "file": "desktop.cso",
                        "bytes": 7,
                        "sha256": corpus.sha256(root / "desktop.cso"),
                    },
                    "dxil": {
                        "original_triple": "dxil-ms-xb2",
                        "target_triple": "dxil-ms-dx",
                        "removed_named_metadata": ["xbox.compiler.options"],
                    },
                }
            ],
        }
        path = root / "corpus.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        check(corpus.main(["verify", "--manifest", str(path)]) == 0, "valid manifest")

        manifest["cases"][0]["xbox"]["sha256"] = "1" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")
        check(corpus.main(["verify", "--manifest", str(path)]) == 1, "hash mismatch")

        manifest["cases"][0]["xbox"]["sha256"] = corpus.sha256(root / "xbox.cso")
        manifest["cases"][0]["dxil"]["target_triple"] = "dxil-ms-xb2"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        check(corpus.main(["verify", "--manifest", str(path)]) == 1, "triple mismatch")

        manifest["cases"][0]["dxil"]["target_triple"] = "dxil-ms-dx"
        (root / "xbox.cso").unlink()
        path.write_text(json.dumps(manifest), encoding="utf-8")
        check(corpus.main(["verify", "--manifest", str(path)]) == 1, "missing file")
    print("verify: hashes, sizes, triples and missing files passed")


def main():
    check_loading()
    check_build_skip()
    check_verify()
    print("gdk_shader_corpus_check: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
