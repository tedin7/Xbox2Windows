#!/usr/bin/env python3
"""Compile an original HLSL corpus with the GDK dxc and retarget it.

The Xbox container and the desktop container stay outside the repository. The
HLSL sources are original and versioned; only SHA-256 values and sizes are
recorded in the manifest.

Commands:
  build --gdk-root <GXDK> --corpus <dir> --out <dir> [--dxc <path>] [--require]
  verify --manifest <corpus.json> [--root <dir>]

build returns 77 when dxc is missing, unless --require is set. verify needs no
SDK and checks the recorded hashes and the DXIL triples.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP = 77


def sha256(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest().upper()


def load_cases(corpus):
    corpus = Path(corpus)
    manifest = corpus / "cases.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    cases = document.get("cases", [])
    names = set()
    for case in cases:
        name = case["name"]
        if name in names:
            raise ValueError("duplicate corpus case: " + name)
        names.add(name)
        source = corpus / case["source"]
        if not source.exists():
            raise ValueError("missing corpus source: " + str(source))
    return cases


def find_dxc(gdk_root, explicit):
    if explicit:
        return Path(explicit) if Path(explicit).exists() else None
    for relative in ("GXDK/bin/Scarlett/dxc.exe", "bin/Scarlett/dxc.exe", "dxc.exe", "dxc"):
        candidate = Path(gdk_root) / relative
        if candidate.exists():
            return candidate
    found = shutil.which("dxc")
    return Path(found) if found else None


def dxc_version(dxc):
    result = subprocess.run([str(dxc), "--version"], capture_output=True, text=True, timeout=60)
    return (result.stdout + result.stderr).strip()


def compile_case(case, corpus, out, dxc):
    directory = out / case["name"]
    directory.mkdir(parents=True, exist_ok=True)
    source = (corpus / case["source"]).resolve()
    xbox = directory / "xbox.cso"
    desktop = directory / "desktop.cso"
    args = [
        str(dxc),
        "-noprecompile",
        "-T",
        case["profile"],
        "-E",
        case["entry"],
    ] + list(case.get("args", [])) + [str(source), "-Fo", str(xbox)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return {"name": case["name"], "error": (result.stdout + result.stderr).strip()}
    retarget = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "dxil_retarget.py"), str(xbox), str(desktop)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if retarget.returncode != 0:
        return {"name": case["name"], "error": (retarget.stdout + retarget.stderr).strip()}
    try:
        dxil = json.loads(retarget.stdout)
    except json.JSONDecodeError:
        dxil = {"raw": retarget.stdout.strip()}
    return {
        "name": case["name"],
        "source": case["source"],
        "source_sha256": sha256(source),
        "profile": case["profile"],
        "entry": case["entry"],
        "args": list(case.get("args", [])),
        "xbox": {
            "file": str(xbox.relative_to(out)),
            "bytes": xbox.stat().st_size,
            "sha256": sha256(xbox),
        },
        "desktop": {
            "file": str(desktop.relative_to(out)),
            "bytes": desktop.stat().st_size,
            "sha256": sha256(desktop),
        },
        "dxil": dxil.get("dxil", dxil),
    }


def build_command(args):
    corpus = Path(args.corpus)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(corpus)
    dxc = find_dxc(args.gdk_root, args.dxc)
    if dxc is None:
        print("dxc not found; skipping the GDK shader corpus", file=sys.stderr)
        return 2 if args.require else SKIP
    version = dxc_version(dxc)
    results = []
    failures = []
    for case in cases:
        result = compile_case(case, corpus, out, dxc)
        results.append(result)
        if "error" in result:
            failures.append(result)
            print(f"{case['name']}: failed", file=sys.stderr)
            if not args.keep_going:
                break
        else:
            print(f"{case['name']}: {result['xbox']['bytes']} -> {result['desktop']['bytes']} bytes")
    document = {
        "sdk": "261000",
        "dxc": version,
        "cases": results,
    }
    (out / "corpus.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(results), "failures": len(failures)}))
    return 2 if failures else 0


def verify_command(args):
    manifest = Path(args.manifest)
    root = Path(args.root) if args.root else manifest.parent
    document = json.loads(manifest.read_text(encoding="utf-8"))
    errors = []
    for case in document.get("cases", []):
        if "error" in case:
            errors.append(case["name"] + ": build error recorded")
            continue
        for kind in ("xbox", "desktop"):
            record = case.get(kind)
            if not record:
                errors.append(f"{case['name']}: missing {kind} record")
                continue
            path = root / record["file"]
            if not path.exists():
                errors.append(f"{case['name']}: missing {path}")
                continue
            if sha256(path) != record["sha256"]:
                errors.append(f"{case['name']}: {kind} hash mismatch")
            if path.stat().st_size != record["bytes"]:
                errors.append(f"{case['name']}: {kind} size mismatch")
        dxil = case.get("dxil", {})
        if dxil.get("original_triple") != "dxil-ms-xb2":
            errors.append(f"{case['name']}: source triple {dxil.get('original_triple')!r}")
        if dxil.get("target_triple") != "dxil-ms-dx":
            errors.append(f"{case['name']}: target triple {dxil.get('target_triple')!r}")
        removed = dxil.get("removed_named_metadata", [])
        if "xbox.compiler.options" not in removed:
            errors.append(f"{case['name']}: xbox.compiler.options not removed")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"verified {len(document.get('cases', []))} cases")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--gdk-root", required=True)
    build.add_argument("--corpus", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--dxc")
    build.add_argument("--require", action="store_true")
    build.add_argument("--keep-going", action="store_true")
    build.set_defaults(func=build_command)

    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--root")
    verify.set_defaults(func=verify_command)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
