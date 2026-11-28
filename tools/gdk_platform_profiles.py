#!/usr/bin/env python3
"""Extract GDK MSBuild platform properties into JSON.

The GDK installation stays outside the repository. This reads the canonical
VS2022/flatDeployment platform files, keeps the MSBuild conditions and resolves
the simple variables defined in the same files. Unresolved variables remain
visible instead of being guessed.

Commands:
  list --root <GDK_ROOT> [--vs v170] [--sdk 261000]
  scan --root <GDK_ROOT> --platform <name> --json <out> [--vs v170] [--sdk 261000]
"""

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


NS = "{http://schemas.microsoft.com/developer/msbuild/2003}"
VARIABLE = re.compile(r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)")
PROPERTY_NAMES = ("PropertyGroup", "ItemDefinitionGroup", "ItemGroup")


def tag(element):
    return element.tag.rsplit("}", 1)[-1]


def sha256(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest().upper()


def platform_dirs(root):
    root = Path(root)
    found = {}
    for candidate in root.rglob("Platform.props"):
        directory = candidate.parent
        found[str(directory).lower()] = directory
    return list(found.values())


def score(directory, vs, sdk):
    parts = [part.lower() for part in directory.parts]
    value = 0
    if "vs2022" in parts:
        value += 20
    if vs.lower() in parts:
        value += 10
    if "flatdeployment" in parts:
        value += 5
    if sdk in parts:
        value += 3
    if (directory / sdk).is_dir():
        value += 15
    return value


def choose(root, platform, vs, sdk):
    matches = [item for item in platform_dirs(root) if item.name.lower() == platform.lower()]
    if not matches:
        return None
    versioned = [item for item in matches if (item / sdk).is_dir()]
    if not versioned:
        return None
    return max(versioned, key=lambda item: (score(item, vs, sdk), -len(str(item))))


def list_command(args):
    rows = []
    by_name = {}
    for directory in platform_dirs(args.root):
        candidate = {
            "platform": directory.name,
            "path": str(directory),
            "score": score(directory, args.vs, args.sdk),
        }
        key = directory.name.lower()
        if key not in by_name or candidate["score"] > by_name[key]["score"]:
            by_name[key] = candidate
    rows = list(by_name.values())
    rows.sort(key=lambda item: item["platform"].lower())
    for row in rows:
        print(f"{row['platform']}\t{row['path']}")
    return 0


def parse_file(path, properties, items, sources):
    sources.append({"path": str(path), "sha256": sha256(path)})
    tree = ET.parse(path)
    root = tree.getroot()
    for group in root:
        name = tag(group)
        condition = group.get("Condition", "")
        if name == "PropertyGroup":
            for child in group:
                key = tag(child)
                value = (child.text or "").strip()
                properties.setdefault(key, []).append(
                    {
                        "value": value,
                        "condition": child.get("Condition", condition),
                        "file": Path(path).name,
                    }
                )
        elif name == "ItemDefinitionGroup":
            for item in group:
                item_name = tag(item)
                bucket = items.setdefault(item_name, {})
                for child in item:
                    key = tag(child)
                    bucket.setdefault(key, []).append(
                        {
                            "value": (child.text or "").strip(),
                            "condition": child.get("Condition", condition),
                        }
                    )


def table(properties):
    result = {}
    for key, entries in properties.items():
        value = entries[-1]["value"]
        if f"$({key})" in value:
            continue
        result[key] = value
    return result


def resolve(text, variables, depth=0, seen=None):
    if depth > 8 or not text:
        return text
    seen = set() if seen is None else seen

    def replace(match):
        name = match.group(1)
        if name in seen:
            return match.group(0)
        value = variables.get(name)
        if value is None:
            return match.group(0)
        return resolve(value, variables, depth + 1, seen | {name})

    replaced = VARIABLE.sub(replace, text)
    if len(replaced) > 16384:
        return replaced[:16384]
    if replaced == text:
        return text
    return resolve(replaced, variables, depth + 1, seen)


def split_list(text):
    if not text:
        return []
    values = []
    for item in text.split(";"):
        item = item.strip()
        if not item or item.startswith("%("):
            continue
        values.append(item)
    return values


def last_item(items, group, key):
    entries = items.get(group, {}).get(key, [])
    return entries[-1]["value"] if entries else ""


def combined(items, group, key, separator):
    entries = items.get(group, {}).get(key, [])
    return separator.join(entry["value"] for entry in entries if entry["value"])


def scan_command(args):
    directory = choose(args.root, args.platform, args.vs, args.sdk)
    if directory is None:
        print(
            f"platform not found for SDK {args.sdk}: {args.platform}",
            file=sys.stderr,
        )
        return 2
    sources = []
    properties = {}
    items = {}
    edition = directory / args.sdk
    files = [
        directory / "Platform.default.props",
        edition / "Platform.Common.props",
        edition / "Platform.Edition.default.props",
        edition / "Platform.Edition.props",
        edition / "Platform.Edition.targets",
        edition / "Platform.Commandline.targets",
        directory / "Platform.props",
        directory / "Platform.targets",
    ]
    toolset_root = directory / "PlatformToolsets"
    if toolset_root.exists():
        files += sorted(toolset_root.glob("*/Toolset.props"))
    for path in files:
        if path.exists():
            parse_file(path, properties, items, sources)
    variables = table(properties)

    def resolved(value):
        return resolve(value, variables)

    defines = split_list(resolved(combined(items, "ClCompile", "PreprocessorDefinitions", ";")))
    includes = split_list(
        resolved(combined(items, "ClCompile", "AdditionalIncludeDirectories", ";"))
    )
    if not includes:
        includes = split_list(resolved(variables.get("Console_EndpointIncludeRoot", "")))
    libs = split_list(resolved(combined(items, "Link", "AdditionalLibraryDirectories", ";")))
    if not libs:
        libs = split_list(resolved(variables.get("Console_EndpointLibRoot", "")))
    dependencies = split_list(
        resolved(combined(items, "Link", "AdditionalDependencies", ";"))
    )
    if not dependencies:
        dependencies = split_list(resolved(variables.get("Console_Libs", "")))

    document = {
        "sdk": args.sdk,
        "platform": directory.name,
        "toolset": args.vs,
        "root": str(directory),
        "sources": sources,
        "defines": defines,
        "includes": includes,
        "libs": libs,
        "dependencies": dependencies,
        "console": {
            "endpoint_include_root": resolved(variables.get("Console_EndpointIncludeRoot", "")),
            "endpoint_lib_root": resolved(variables.get("Console_EndpointLibRoot", "")),
            "libs": split_list(resolved(variables.get("Console_Libs", ""))),
            "tool_path": resolved(variables.get("Console_SdkToolPath", "")),
            "sdk_root": resolved(variables.get("Console_SdkRoot", "")),
        },
        "flags": {
            "additional_options": split_list(
                resolved(combined(items, "ClCompile", "AdditionalOptions", " "))
            ),
            "ignore_specific_default_libraries": split_list(
                resolved(last_item(items, "Link", "IgnoreSpecificDefaultLibraries"))
            ),
            "runtime_library": resolved(last_item(items, "ClCompile", "RuntimeLibrary")),
            "exception_handling": resolved(last_item(items, "ClCompile", "ExceptionHandling")),
            "language_standard": resolved(last_item(items, "ClCompile", "LanguageStandard")),
            "floating_point_model": resolved(
                last_item(items, "ClCompile", "FloatingPointModel")
            ),
            "enable_enhanced_instruction_set": resolved(
                last_item(items, "ClCompile", "EnableEnhancedInstructionSet")
            ),
            "target_machine": resolved(last_item(items, "Link", "TargetMachine")),
            "use_debug_libraries": resolved(variables.get("UseDebugLibraries", "")),
            "character_set": resolved(variables.get("CharacterSet", "")),
        },
        "properties": {
            key: [
                {"value": entry["value"], "condition": entry["condition"], "file": entry["file"]}
                for entry in entries
            ]
            for key, entries in properties.items()
        },
        "items": items,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "platform": document["platform"],
                "sources": len(sources),
                "defines": len(document["defines"]),
                "includes": len(document["includes"]),
                "libs": len(document["libs"]),
                "dependencies": len(document["dependencies"]),
            }
        )
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "scan"):
        command = sub.add_parser(name)
        command.add_argument("--root", required=True)
        command.add_argument("--vs", default="v170")
        command.add_argument("--sdk", default="261000")
        if name == "scan":
            command.add_argument("--platform", required=True)
            command.add_argument("--json", required=True)
            command.set_defaults(func=scan_command)
        else:
            command.set_defaults(func=list_command)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
