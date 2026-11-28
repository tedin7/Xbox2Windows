#!/usr/bin/env python3
"""Merge GDK knowledge facts, report conflicts and count coverage.

The tool reads only JSON metadata. It never loads a GDK binary, header or SDK
file. Inputs are optional directories under a knowledge root:

    layouts/*.json       layout classes with a size and members
    abi/*.json           ABI facts (structs, enums, constants, GUIDs)
    exports/*.json       export rows, or a root exports-all.json
    behavior/*.json      runtime observations
    platform/*.json      build defines, includes, libraries and flags
    shaders/corpus.json  shader corpus cases

Normalised facts carry module, kind, name, value, source, confidence and
targets. Conflicts are grouped by fact identity and by type name. The report is
deterministic: facts, conflicts, modules and shims are sorted before writing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from collections import defaultdict


SCHEMA = "xbox2windows.gdk_knowledge/1"
TOOL = "gdk_knowledge_report"
VERSION = 1
CATEGORIES = ("layouts", "abi", "exports", "behavior", "platform", "shaders")
TYPE_KINDS = frozenset(
    {"layout", "class", "struct", "union", "type", "typedef", "sizeof"}
)
ENUM_KINDS = frozenset({"enum", "enumerator", "enumeration", "enum-value"})
CONFLICT_ORDER = {"value": 0, "size": 1, "ordinal": 2, "enum": 3, "type-size": 4}
FACT_FIELDS = ("module", "kind", "name", "value", "source", "confidence", "targets")
DEFAULT_CONFIDENCE = {
    "layouts": "header",
    "abi": "compiled",
    "exports": "export-table",
    "behavior": "runtime",
    "platform": "build-profile",
    "shaders": "shader-corpus",
}
LEFTOVER_KEYS = frozenset(
    {"module", "kind", "name", "value", "source", "confidence", "targets"}
)
METHOD_SOURCES = frozenset(
    {
        "msvc-layout",
        "compiled-probe",
        "dumpbin",
        "runtime-oracle",
        "header-msvc-layout",
    }
)


class InputError(Exception):
    """A malformed input or unusable path. The CLI maps this to exit code 2."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _as_int(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.lower().startswith(("0x", "-0x")):
                return int(text, 16)
            return int(text, 10)
        except ValueError:
            return None
    return None


def _display_source(source):
    if isinstance(source, str):
        return source
    return _canonical(source)


def _value_sort_key(value):
    if isinstance(value, bool):
        return (0, int(value), "")
    if isinstance(value, (int, float)):
        return (1, value, "")
    if isinstance(value, str):
        return (2, 0, value)
    return (3, 0, _canonical(value))


def _relative(path, root):
    path = Path(path)
    root = Path(root)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.name


def _read_json(path):
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise InputError("cannot read {}: {}".format(path, exc)) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(
            "invalid JSON in {} at line {} column {}: {}".format(
                path, exc.lineno, exc.colno, exc.msg
            )
        ) from exc


def _require_object(value, where):
    if not isinstance(value, dict):
        raise InputError("{} must be a JSON object".format(where))
    return value


def _require_list(value, where):
    if not isinstance(value, list):
        raise InputError("{} must be a JSON array".format(where))
    return value


def _require_name(item, where, keys=("name",)):
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise InputError("{} has no non-empty {}".format(where, "/".join(keys)))


def _normalise_targets(value, where):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise InputError("{} targets must be a string or an array".format(where))
    targets = []
    for target in value:
        if not isinstance(target, str):
            raise InputError("{} targets must contain strings".format(where))
        if target:
            targets.append(target)
    return sorted(set(targets))


def _source_and_confidence(item, meta, default_source, default_confidence):
    item = item if isinstance(item, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    source = None
    method = None
    for candidate in (item.get("source"), meta.get("source")):
        if isinstance(candidate, str) and candidate.strip():
            text = candidate.strip()
            if text.casefold() in METHOD_SOURCES:
                if method is None:
                    method = text
            elif source is None:
                source = text
        elif candidate is not None and source is None:
            source = candidate
    confidence = (
        item.get("confidence")
        or meta.get("confidence")
        or method
        or default_confidence
    )
    return source or default_source, confidence


def _module_from_stem(stem):
    text = Path(stem).stem
    lowered = text.casefold()
    suffixes = (
        "-all-layout",
        "-interesting-layout",
        "-layout-report",
        "-layout",
        "_layout",
        "-vtables",
        "-facts",
        "-exports",
    )
    for suffix in suffixes:
        if lowered.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return text


def _module_from_library(path):
    name = Path(path).name
    lowered = name.casefold()
    for suffix in (".lib", ".dll"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return _module_from_stem(name)


def _module_from_def_stem(stem):
    module = _module_from_stem(stem)
    lowered = module.casefold()
    for suffix in ("_shim", "-shim"):
        if lowered.endswith(suffix) and len(module) > len(suffix):
            return module[: -len(suffix)]
    return module


def _make_fact(module, kind, name, value, source, confidence, targets):
    if not isinstance(module, str) or not module.strip():
        raise InputError("fact {} has no module".format(name))
    if not isinstance(kind, str) or not kind.strip():
        raise InputError("fact {} has no kind".format(name))
    if not isinstance(name, str) or not name.strip():
        raise InputError("fact has no non-empty name")
    if confidence is None:
        confidence = ""
    if not isinstance(confidence, str):
        confidence = str(confidence)
    return {
        "module": module.strip(),
        "kind": kind.strip(),
        "name": name.strip(),
        "value": value,
        "source": source,
        "confidence": confidence,
        "targets": list(targets),
    }


def _fact_sort_key(fact):
    return (
        fact["module"].casefold(),
        fact["module"],
        fact["kind"].casefold(),
        fact["kind"],
        fact["name"].casefold(),
        fact["name"],
        _canonical(fact["value"]),
        _canonical(fact["source"]),
    )


def _dedupe_facts(facts):
    seen = set()
    result = []
    for fact in facts:
        key = _canonical(fact)
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    result.sort(key=_fact_sort_key)
    return result


def _canonicalise_modules(facts):
    counts = defaultdict(lambda: defaultdict(int))
    for fact in facts:
        counts[fact["module"].casefold()][fact["module"]] += 1
    canonical = {}
    for key, variants in counts.items():
        if len(variants) == 1:
            canonical[key] = next(iter(variants))
            continue
        canonical[key] = sorted(
            variants,
            key=lambda name: (
                -variants[name],
                name != name.lower(),
                name.casefold(),
                name,
            ),
        )[0]
    for fact in facts:
        fact["module"] = canonical[fact["module"].casefold()]
    return facts


def _is_type_kind(kind):
    return kind.casefold() in TYPE_KINDS


def _is_enum_kind(kind):
    lowered = kind.casefold().replace("_", "-")
    return lowered in ENUM_KINDS or lowered.startswith("enum-")


def _is_anonymous_name(name):
    lowered = name.casefold()
    return "<unnamed" in lowered or "<anonymous" in lowered


def _size_of(fact):
    if not _is_type_kind(fact["kind"]):
        return None
    value = fact["value"]
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        for key in ("size", "sizeof", "bytes", "size_bytes"):
            if key in value:
                number = _as_int(value[key])
                if number is not None:
                    return number
    return None


def _ordinal_of(fact):
    value = fact["value"]
    if isinstance(value, dict):
        return _as_int(value.get("ordinal"))
    return _as_int(value)


def _json_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".json"),
        key=lambda path: path.name.casefold(),
    )


def _load_layouts(path, root):
    data = _read_json(path)
    where = _relative(path, root)
    meta = data if isinstance(data, dict) else {}
    if isinstance(data, list):
        classes = data
    elif isinstance(data, dict):
        classes = data.get("classes")
        if classes is None and "name" in data:
            classes = [data]
        if classes is None:
            raise InputError("{} has no classes array".format(where))
        _require_list(classes, where + " classes")
    else:
        raise InputError("{} must contain an object or an array".format(where))

    fallback_module = _module_from_stem(path.stem)
    top_module = meta.get("module")
    facts = []
    for index, item in enumerate(classes):
        item_where = "{} classes[{}]".format(where, index)
        item = _require_object(item, item_where)
        name = _require_name(item, item_where)
        size = _as_int(item.get("size"))
        if size is None:
            size = _as_int(item.get("bytes"))
        if size is None or size < 0:
            raise InputError("{} has no non-negative integer size".format(item_where))
        members = item.get("members", [])
        if members is None:
            members = []
        _require_list(members, item_where + " members")
        module = item.get("module") or top_module or fallback_module
        kind = item.get("kind") or "layout"
        source, confidence = _source_and_confidence(
            item, meta, where, DEFAULT_CONFIDENCE["layouts"]
        )
        targets = _normalise_targets(
            item["targets"] if "targets" in item else meta.get("targets"),
            item_where,
        )
        facts.append(
            _make_fact(
                module,
                kind,
                name,
                {"size": size, "members": members},
                source,
                confidence,
                targets,
            )
        )
    return facts


def _load_abi(path, root):
    data = _read_json(path)
    where = _relative(path, root)
    if isinstance(data, list):
        facts_data = data
        meta = {}
    elif isinstance(data, dict):
        meta = data
        facts_data = data.get("facts")
        if facts_data is None:
            raise InputError("{} has no facts array".format(where))
        _require_list(facts_data, where + " facts")
    else:
        raise InputError("{} must contain an object or an array".format(where))

    fallback_module = _module_from_stem(path.stem)
    top_module = meta.get("module")
    facts = []
    for index, item in enumerate(facts_data):
        item_where = "{} facts[{}]".format(where, index)
        item = _require_object(item, item_where)
        name = _require_name(item, item_where)
        kind = item.get("kind") or meta.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise InputError("{} has no non-empty kind".format(item_where))
        if "value" in item:
            value = item["value"]
        else:
            value = {key: val for key, val in item.items() if key not in LEFTOVER_KEYS}
        module = item.get("module") or top_module or fallback_module
        source, confidence = _source_and_confidence(
            item, meta, where, DEFAULT_CONFIDENCE["abi"]
        )
        targets = _normalise_targets(
            item["targets"] if "targets" in item else meta.get("targets"),
            item_where,
        )
        facts.append(
            _make_fact(module, kind.strip(), name, value, source, confidence, targets)
        )
    return facts


def _export_groups(data, where):
    if isinstance(data, list):
        if not data:
            return []
        if all(isinstance(item, dict) and isinstance(item.get("exports"), list) for item in data):
            return data
        if all(isinstance(item, dict) and "name" in item for item in data):
            return [{"exports": data}]
        raise InputError("{} export array has an unsupported shape".format(where))

    data = _require_object(data, where)
    if isinstance(data.get("modules"), list):
        return _require_list(data["modules"], where + " modules")
    if isinstance(data.get("modules"), dict):
        groups = []
        for module, rows in data["modules"].items():
            if not isinstance(rows, list):
                raise InputError(
                    "{} module {} must have an exports array".format(where, module)
                )
            groups.append({"module": module, "exports": rows})
        return groups
    if "exports" in data:
        return [data]

    metadata = {
        "schema",
        "generated",
        "sdk",
        "module",
        "source",
        "confidence",
        "targets",
        "kind",
        "file",
    }
    groups = []
    for module, rows in data.items():
        if module in metadata:
            continue
        if not isinstance(rows, list):
            raise InputError("{} mapping entry {} must be an array".format(where, module))
        groups.append({"module": module, "exports": rows})
    if not groups:
        raise InputError("{} has no exports array".format(where))
    return groups


def _load_exports(path, root):
    data = _read_json(path)
    where = _relative(path, root)
    groups = _export_groups(data, where)
    meta = data if isinstance(data, dict) else {}
    fallback_module = _module_from_stem(path.stem)
    fallback_source = where
    top_module = meta.get("module")

    facts = []
    for group_index, group in enumerate(groups):
        group_where = "{} exports[{}]".format(where, group_index)
        group = _require_object(group, group_where)
        rows = group.get("exports", [])
        _require_list(rows, group_where + " exports")

        module = group.get("module") or top_module
        if not module and group.get("file"):
            module = _module_from_library(group["file"])
        if not module:
            module = fallback_module

        source, confidence = _source_and_confidence(
            group, meta, fallback_source, DEFAULT_CONFIDENCE["exports"]
        )
        if group.get("file"):
            origin = Path(str(group["file"])).name
            if origin:
                source = "{}#{}".format(source, origin)
        targets = _normalise_targets(
            group["targets"] if "targets" in group else meta.get("targets"),
            group_where,
        )

        for row_index, row in enumerate(rows):
            row_where = "{} row {}".format(group_where, row_index)
            if isinstance(row, str):
                row = {"name": row}
            row = _require_object(row, row_where)
            name = _require_name(row, row_where)
            ordinal = _as_int(row.get("ordinal"))
            rva = row.get("rva")
            parsed_rva = _as_int(rva)
            if parsed_rva is not None:
                rva = parsed_rva
            row_module = row.get("module") or module
            row_source = row.get("source") or source
            row_confidence = row.get("confidence") or confidence
            row_targets = _normalise_targets(
                row["targets"] if "targets" in row else targets,
                row_where,
            )
            facts.append(
                _make_fact(
                    row_module,
                    "export",
                    name,
                    {"ordinal": ordinal, "rva": rva},
                    row_source,
                    row_confidence,
                    row_targets,
                )
            )
    return facts


def _observation_value(item):
    payload = {
        key: value
        for key, value in item.items()
        if key not in {"module", "kind", "name", "source", "confidence", "targets"}
    }
    observation_kind = item.get("kind")
    if isinstance(observation_kind, str):
        lowered = observation_kind.casefold()
        if lowered not in {"behavior", "runtime", "observation", "value", ""}:
            payload["observation"] = observation_kind
    if not payload:
        return None
    if set(payload) == {"value"}:
        return payload["value"]
    return payload


def _load_behavior(path, root):
    data = _read_json(path)
    where = _relative(path, root)
    if isinstance(data, list):
        observations = data
        meta = {}
    elif isinstance(data, dict):
        meta = data
        observations = data.get("observations")
        if observations is None:
            raise InputError("{} has no observations array".format(where))
        _require_list(observations, where + " observations")
    else:
        raise InputError("{} must contain an object or an array".format(where))

    fallback_module = _module_from_stem(path.stem)
    top_module = meta.get("module")
    facts = []
    for index, item in enumerate(observations):
        item_where = "{} observations[{}]".format(where, index)
        item = _require_object(item, item_where)
        name = _require_name(item, item_where)
        module = item.get("module") or top_module or fallback_module
        source, confidence = _source_and_confidence(
            item, meta, where, DEFAULT_CONFIDENCE["behavior"]
        )
        targets = _normalise_targets(
            item["targets"] if "targets" in item else meta.get("targets"),
            item_where,
        )
        facts.append(
            _make_fact(
                module,
                "behavior",
                name,
                _observation_value(item),
                source,
                confidence,
                targets,
            )
        )
    return facts


def _load_platform(path, root):
    data = _read_json(path)
    where = _relative(path, root)
    if isinstance(data, list):
        items = data
        meta = {}
    elif isinstance(data, dict):
        meta = data
        platforms = data.get("platforms")
        if platforms is not None:
            _require_list(platforms, where + " platforms")
            items = platforms
        else:
            items = [data]
    else:
        raise InputError("{} must contain an object or an array".format(where))

    fallback_module = _module_from_stem(path.stem)
    top_module = meta.get("module")
    facts = []
    for index, item in enumerate(items):
        item_where = "{} platforms[{}]".format(where, index)
        item = _require_object(item, item_where)
        values = {}
        has_values = False
        for key in ("defines", "includes", "libs", "flags"):
            rows = item.get(key, [])
            if rows is None:
                rows = {} if key == "flags" else []
            if isinstance(rows, str):
                rows = [rows]
            if not isinstance(rows, (list, dict)):
                raise InputError(
                    "{} {} must be an array or object".format(item_where, key)
                )
            values[key] = rows
            has_values = has_values or bool(rows)
        name = item.get("platform") or item.get("name") or item.get("id")
        if not name and not has_values and not item.get("module"):
            raise InputError("{} has no platform name or build facts".format(item_where))
        module = item.get("module") or top_module or name or fallback_module
        if not name:
            name = module
        if not isinstance(name, str) or not name.strip():
            raise InputError("{} has no non-empty platform name".format(item_where))
        source, confidence = _source_and_confidence(
            item, meta, where, DEFAULT_CONFIDENCE["platform"]
        )
        targets = _normalise_targets(
            item["targets"] if "targets" in item else meta.get("targets"),
            item_where,
        )
        facts.append(
            _make_fact(
                module,
                "platform",
                name,
                values,
                source,
                confidence,
                targets,
            )
        )
    return facts


def _load_shaders(path, root):
    data = _read_json(path)
    where = _relative(path, root)
    if isinstance(data, list):
        cases = data
        meta = {}
    elif isinstance(data, dict):
        meta = data
        cases = data.get("cases")
        if cases is None:
            raise InputError("{} has no cases array".format(where))
        _require_list(cases, where + " cases")
    else:
        raise InputError("{} must contain an object or an array".format(where))

    fallback_module = meta.get("module") or meta.get("platform") or path.parent.name
    facts = []
    for index, item in enumerate(cases):
        item_where = "{} cases[{}]".format(where, index)
        item = _require_object(item, item_where)
        name = None
        for key in ("name", "id", "case"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                name = value.strip()
                break
        if name is None:
            name = "case-{:04d}".format(index)
        value = {
            key: val
            for key, val in item.items()
            if key not in {"module", "name", "id", "case", "source", "confidence", "targets"}
        }
        module = item.get("module") or fallback_module
        source, confidence = _source_and_confidence(
            item, meta, where, DEFAULT_CONFIDENCE["shaders"]
        )
        targets = _normalise_targets(
            item["targets"] if "targets" in item else meta.get("targets"),
            item_where,
        )
        facts.append(
            _make_fact(module, "shader", name, value, source, confidence, targets)
        )
    return facts


def load_knowledge(root):
    root = Path(root)
    if not root.exists():
        raise InputError("knowledge root does not exist: {}".format(root))
    if not root.is_dir():
        raise InputError("knowledge root is not a directory: {}".format(root))

    category_files = {category: [] for category in CATEGORIES}
    category_facts = {category: 0 for category in CATEGORIES}
    facts = []

    for category in ("layouts", "abi", "behavior", "platform"):
        loader = {
            "layouts": _load_layouts,
            "abi": _load_abi,
            "behavior": _load_behavior,
            "platform": _load_platform,
        }[category]
        files = _json_files(root / category)
        for path in files:
            loaded = loader(path, root)
            facts.extend(loaded)
            category_facts[category] += len(loaded)
        category_files[category] = [_relative(path, root) for path in files]

    export_files = _json_files(root / "exports")
    if not export_files:
        root_exports = root / "exports-all.json"
        if root_exports.is_file():
            export_files = [root_exports]
    for path in export_files:
        loaded = _load_exports(path, root)
        facts.extend(loaded)
        category_facts["exports"] += len(loaded)
    category_files["exports"] = [_relative(path, root) for path in export_files]

    shader_corpus = root / "shaders" / "corpus.json"
    if not shader_corpus.is_file():
        shader_corpus = root / "shaders" / "corpus" / "corpus.json"
    if shader_corpus.is_file():
        loaded = _load_shaders(shader_corpus, root)
        facts.extend(loaded)
        category_facts["shaders"] = len(loaded)
        category_files["shaders"] = [_relative(shader_corpus, root)]

    return facts, category_files, category_facts


def _unique_value_entries(entries):
    chosen = {}
    for fact, value in entries:
        key = _canonical(value)
        source = _display_source(fact["source"])
        candidate = (fact["module"], value, source)
        previous = chosen.get(key)
        if previous is None or (source.casefold(), source) < (
            previous[2].casefold(),
            previous[2],
        ):
            chosen[key] = candidate
    return [
        {"module": module, "value": value, "source": source}
        for module, value, source in sorted(
            chosen.values(),
            key=lambda item: (
                item[0].casefold(),
                item[0],
                _value_sort_key(item[1]),
                item[2].casefold(),
                item[2],
            ),
        )
    ]


def _fact_value_entries(items):
    return _unique_value_entries([(fact, fact["value"]) for fact in items])


def _conflict(kind, module, name, values, message, modules=None):
    sources = sorted({_display_source(item["source"]) for item in values})
    conflict = {
        "kind": kind,
        "module": module,
        "name": name,
        "values": values,
        "sources": sources,
        "message": message,
    }
    if modules is not None:
        conflict["modules"] = sorted(modules)
    return conflict


def detect_conflicts(facts):
    grouped = defaultdict(list)
    type_by_module_name = defaultdict(list)
    type_by_name = defaultdict(lambda: defaultdict(list))
    enum_by_name = defaultdict(list)
    export_by_name = defaultdict(list)

    for fact in facts:
        grouped[(fact["module"], fact["kind"], fact["name"])].append(fact)
        kind = fact["kind"]
        anonymous_type = _is_type_kind(kind) and _is_anonymous_name(fact["name"])
        if _is_type_kind(kind) and not anonymous_type:
            size = _size_of(fact)
            if size is not None:
                type_by_module_name[(fact["module"], fact["name"])].append((fact, size))
                type_by_name[fact["name"]][fact["module"]].append((fact, size))
        if _is_enum_kind(kind) and not _is_anonymous_name(fact["name"]):
            enum_by_name[(fact["module"], fact["name"])].append(fact)
        if kind.casefold() == "export":
            export_by_name[(fact["module"], fact["name"])].append(fact)

    conflicts = []
    size_conflict_keys = set()
    for (module, name), entries in sorted(type_by_module_name.items()):
        sizes = {size for _, size in entries}
        if len(sizes) <= 1:
            continue
        values = _unique_value_entries(entries)
        conflicts.append(
            _conflict(
                "size",
                module,
                name,
                values,
                "type {} has conflicting sizes in module {}: {}".format(
                    name, module, ", ".join(str(size) for size in sorted(sizes))
                ),
            )
        )
        size_conflict_keys.add((module, name))

    enum_conflict_keys = set()
    for (module, name), items in sorted(enum_by_name.items()):
        values = _fact_value_entries(items)
        if len(values) <= 1:
            continue
        conflicts.append(
            _conflict(
                "enum",
                module,
                name,
                values,
                "enum {} has conflicting values in module {}".format(name, module),
            )
        )
        enum_conflict_keys.add((module, name))

    for (module, name), items in sorted(export_by_name.items()):
        ordinals = sorted({_ordinal_of(fact) for fact in items if _ordinal_of(fact) is not None})
        if len(ordinals) <= 1:
            continue
        values = [
            {
                "module": module,
                "value": ordinal,
                "source": _display_source(
                    next(
                        fact["source"]
                        for fact in items
                        if _ordinal_of(fact) == ordinal
                    )
                ),
            }
            for ordinal in ordinals
        ]
        conflicts.append(
            _conflict(
                "ordinal",
                module,
                name,
                values,
                "export {} has conflicting ordinals: {}".format(
                    name, ", ".join(str(ordinal) for ordinal in ordinals)
                ),
            )
        )

    for (module, kind, name), items in sorted(grouped.items()):
        values = _fact_value_entries(items)
        if len(values) <= 1:
            continue
        if kind.casefold() == "export":
            continue
        if (_is_type_kind(kind) or _is_enum_kind(kind)) and _is_anonymous_name(name):
            continue
        if _is_type_kind(kind) and (module, name) in size_conflict_keys:
            continue
        if _is_enum_kind(kind) and (module, name) in enum_conflict_keys:
            continue
        conflicts.append(
            _conflict(
                "value",
                module,
                name,
                values,
                "{} {} has {} distinct values in module {}".format(
                    kind, name, len(values), module
                ),
            )
        )

    for name in sorted(type_by_name, key=lambda text: (text.casefold(), text)):
        modules = type_by_name[name]
        if len(modules) < 2:
            continue
        all_sizes = {size for entries in modules.values() for _, size in entries}
        if len(all_sizes) <= 1:
            continue
        values = []
        for module in sorted(modules, key=lambda text: (text.casefold(), text)):
            unique_sizes = {}
            for fact, size in modules[module]:
                key = _canonical(size)
                source = _display_source(fact["source"])
                previous = unique_sizes.get(key)
                if previous is None or (source.casefold(), source) < (
                    previous[0].casefold(),
                    previous[0],
                ):
                    unique_sizes[key] = (source, size)
            for source, size in sorted(
                unique_sizes.values(), key=lambda item: (_value_sort_key(item[1]), item[0])
            ):
                values.append({"module": module, "value": size, "source": source})
        module_summary = ", ".join(
            "{}={}".format(
                module,
                "/".join(
                    str(size)
                    for size in sorted(
                        {size for _, size in modules[module]}
                    )
                ),
            )
            for module in sorted(modules, key=lambda text: (text.casefold(), text))
        )
        conflicts.append(
            _conflict(
                "type-size",
                "*",
                name,
                values,
                "type {} has incompatible sizes across modules: {}".format(
                    name, module_summary
                ),
                modules=modules,
            )
        )

    conflicts.sort(
        key=lambda item: (
            CONFLICT_ORDER.get(item["kind"], 99),
            item["kind"],
            item["module"].casefold(),
            item["module"],
            item["name"].casefold(),
            item["name"],
            _canonical(item["values"]),
        )
    )
    return conflicts


def _is_internal_shim(name):
    return name.casefold().startswith("x2w")


def _clean_export_name(name):
    text = name.strip().strip('"')
    match = re.match(r"^(.*?)@\d+$", text)
    if match:
        text = match.group(1)
    return text.strip()


def _parse_def(path, root):
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    module = None
    in_exports = False
    entries = []
    source = _relative(path, root)

    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("LIBRARY"):
            rest = line[len("LIBRARY"):].strip()
            module = rest.strip().strip('"') or None
            continue
        if upper.startswith("EXPORTS"):
            in_exports = True
            continue
        if not in_exports:
            continue

        token = line.split()[0]
        alias = None
        if "=" in token:
            token, alias = token.split("=", 1)
        export_name = _clean_export_name(token)
        if alias:
            alias = _clean_export_name(alias)
        if not export_name or export_name.startswith(";"):
            continue
        entries.append(
            {
                "module": module or _module_from_def_stem(path.stem),
                "name": export_name,
                "alias": alias,
                "source": source,
            }
        )
    return entries


def _load_shim_mapping(path):
    data = _read_json(path)
    source = path.name
    entries = []
    default_module = data.get("module") if isinstance(data, dict) else None

    def add(module, item):
        if isinstance(item, str):
            name = item.strip()
            alias = None
        elif isinstance(item, dict):
            name = item.get("name")
            alias = item.get("alias")
        else:
            raise InputError("{} shim entry must be a string or object".format(source))
        if not isinstance(name, str) or not name.strip():
            raise InputError("{} shim entry has no non-empty name".format(source))
        if alias is not None:
            if not isinstance(alias, str):
                raise InputError("{} shim alias must be a string".format(source))
            alias = alias.strip() or None
        entries.append(
            {
                "module": module or "shims",
                "name": name.strip(),
                "alias": alias,
                "source": source,
            }
        )

    def add_mapping(mapping):
        for module, rows in mapping.items():
            if not isinstance(rows, list):
                raise InputError("{} mapping {} must be an array".format(source, module))
            for item in rows:
                add(module, item)

    def add_rows(module, rows):
        if not isinstance(rows, list):
            raise InputError("{} group {} must be an array".format(source, module))
        for item in rows:
            add(module, item)

    metadata = {
        "schema",
        "generated",
        "sdk",
        "module",
        "source",
        "confidence",
        "targets",
        "note",
        "version",
    }

    if isinstance(data, dict):
        if isinstance(data.get("modules"), dict):
            add_mapping(data["modules"])
        elif isinstance(data.get("modules"), list):
            for group in data["modules"]:
                group = _require_object(group, source + " module group")
                module = group.get("module") or default_module
                rows = group.get("exports", group.get("shims"))
                if rows is None:
                    raise InputError("{} module group has no exports".format(source))
                add_rows(module, rows)
        elif isinstance(data.get("shims"), list):
            for item in data["shims"]:
                if isinstance(item, dict) and item.get("module"):
                    add(item["module"], item)
                else:
                    add(default_module, item)
        elif isinstance(data.get("shims"), dict):
            add_mapping(data["shims"])
        elif isinstance(data.get("exports"), list):
            add_rows(default_module, data["exports"])
        else:
            mapping = {}
            for key, value in data.items():
                if key in metadata:
                    continue
                if not isinstance(value, list):
                    raise InputError(
                        "{} mapping {} must be an array".format(source, key)
                    )
                mapping[key] = value
            if mapping:
                add_mapping(mapping)
            elif not data:
                return []
            else:
                raise InputError("{} has no shim mapping".format(source))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("exports"), list):
                module = item.get("module") or default_module
                add_rows(module, item["exports"])
            elif isinstance(item, dict) and item.get("module"):
                add(item["module"], item)
            else:
                add(default_module, item)
    else:
        raise InputError("{} must contain a shim mapping".format(source))
    return entries


def load_shims(path):
    if path is None:
        return {
            "source": None,
            "exports": 0,
            "covered": 0,
            "without_fact": [],
            "internal": 0,
        }
    path = Path(path)
    if not path.exists():
        raise InputError("defs path does not exist: {}".format(path))

    entries = []
    if path.is_file():
        if path.suffix.lower() == ".json":
            entries = _load_shim_mapping(path)
        elif path.suffix.lower() == ".def":
            entries = _parse_def(path, path.parent)
        else:
            raise InputError("defs file must be .def or .json: {}".format(path))
    elif path.is_dir():
        files = sorted(
            (
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() == ".def"
            ),
            key=lambda item: item.as_posix().casefold(),
        )
        for item in files:
            entries.extend(_parse_def(item, path))
    else:
        raise InputError("defs path is not a file or directory: {}".format(path))

    merged = {}
    internal = 0
    for entry in entries:
        if _is_internal_shim(entry["name"]):
            internal += 1
            continue
        key = (entry["module"].casefold(), entry["name"].casefold())
        if key not in merged:
            merged[key] = entry
        elif not merged[key].get("alias") and entry.get("alias"):
            merged[key]["alias"] = entry["alias"]

    result = []
    for entry in sorted(
        merged.values(),
        key=lambda item: (
            item["module"].casefold(),
            item["module"],
            item["name"].casefold(),
            item["name"],
        ),
    ):
        result.append(
            {
                "module": entry["module"],
                "name": entry["name"],
                "alias": entry.get("alias"),
                "source": entry["source"],
            }
        )
    return {
        "source": str(path),
        "exports": len(result),
        "covered": 0,
        "without_fact": result,
        "internal": internal,
    }


def _associate_shims(shims, facts):
    if not shims["without_fact"]:
        return shims
    exact = {fact["name"] for fact in facts}
    folded = {fact["name"].casefold() for fact in facts}
    covered = 0
    missing = []
    for entry in shims["without_fact"]:
        names = [entry["name"]]
        if entry.get("alias"):
            names.append(entry["alias"])
        if any(name in exact or name.casefold() in folded for name in names):
            covered += 1
        else:
            missing.append(entry)
    shims["covered"] = covered
    shims["without_fact"] = missing
    return shims


def _module_rows(facts):
    rows = {}
    for fact in facts:
        module = fact["module"]
        row = rows.setdefault(
            module,
            {
                "module": module,
                "types": set(),
                "exports": set(),
                "runtime_facts": 0,
                "shader_facts": 0,
                "platform_facts": 0,
                "facts": 0,
            },
        )
        row["facts"] += 1
        kind = fact["kind"]
        if _is_type_kind(kind):
            row["types"].add(fact["name"])
        elif kind.casefold() == "export":
            row["exports"].add(fact["name"])
        elif kind.casefold() == "behavior":
            row["runtime_facts"] += 1
        elif kind.casefold() == "shader":
            row["shader_facts"] += 1
        elif kind.casefold() == "platform":
            row["platform_facts"] += 1

    result = []
    for module in sorted(rows, key=lambda text: (text.casefold(), text)):
        row = rows[module]
        result.append(
            {
                "module": row["module"],
                "types": len(row["types"]),
                "exports": len(row["exports"]),
                "runtime_facts": row["runtime_facts"],
                "shader_facts": row["shader_facts"],
                "platform_facts": row["platform_facts"],
                "facts": row["facts"],
            }
        )
    return result


def _count_unique_types(facts):
    return len(
        {
            (fact["module"], fact["name"])
            for fact in facts
            if _is_type_kind(fact["kind"])
        }
    )


def _count_unique_exports(facts):
    return len(
        {
            (fact["module"], fact["name"])
            for fact in facts
            if fact["kind"].casefold() == "export"
        }
    )


def build_report(root, defs=None, required=()):
    facts, category_files, category_facts = load_knowledge(root)
    facts = _canonicalise_modules(facts)
    facts = _dedupe_facts(facts)
    conflicts = detect_conflicts(facts)
    shims = _associate_shims(load_shims(defs), facts)

    categories = {}
    missing = []
    for category in CATEGORIES:
        files = category_files[category]
        present = bool(files)
        if not present:
            missing.append(category)
        categories[category] = {
            "present": present,
            "files": files,
            "facts": category_facts[category],
        }

    required = list(required)
    required_missing = [category for category in required if category in missing]

    modules = _module_rows(facts)
    runtime_facts = sum(fact["kind"].casefold() == "behavior" for fact in facts)
    shader_facts = sum(fact["kind"].casefold() == "shader" for fact in facts)
    platforms = len(
        {fact["name"] for fact in facts if fact["kind"].casefold() == "platform"}
    )
    summary = {
        "modules": len(modules),
        "types": _count_unique_types(facts),
        "classes_types": _count_unique_types(facts),
        "exports": _count_unique_exports(facts),
        "export_rows": category_facts["exports"],
        "runtime_facts": runtime_facts,
        "shader_facts": shader_facts,
        "platforms": platforms,
        "facts": len(facts),
        "conflicts": len(conflicts),
        "shims_without_fact": len(shims["without_fact"]),
        "shims": shims["exports"],
    }

    return {
        "schema": SCHEMA,
        "generated_by": {"tool": TOOL, "version": VERSION},
        "inputs": {
            "root": str(root),
            "defs": str(defs) if defs is not None else None,
        },
        "summary": summary,
        "categories": categories,
        "missing": missing,
        "required": required,
        "required_missing": required_missing,
        "modules": modules,
        "conflicts": conflicts,
        "shims": shims,
        "facts": facts,
    }


def _md_cell(value):
    return (
        str(value)
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _md_value(value):
    if isinstance(value, str):
        return value
    return _canonical(value)


def _md_values(values):
    parts = []
    for item in values:
        value = item.get("value")
        if isinstance(value, dict) and "size" in value and "members" in value:
            shown = "size={} members={}".format(value["size"], len(value["members"]))
        else:
            shown = _md_value(value)
        if len(shown) > 160:
            shown = shown[:157] + "..."
        parts.append(
            "{}={} ({})".format(
                item.get("module", ""),
                shown,
                _md_cell(item.get("source", "")),
            )
        )
    return "; ".join(parts)


def render_markdown(report):
    summary = report["summary"]
    lines = [
        "# GDK knowledge coverage",
        "",
        "- Schema: {}".format(report["schema"]),
        "- Tool: {} version {}".format(
            report["generated_by"]["tool"], report["generated_by"]["version"]
        ),
        "- Knowledge root: {}".format(_md_cell(report["inputs"]["root"])),
        "",
        "The report merges metadata facts from the knowledge root. It does not read",
        "or execute SDK binaries, and it does not infer a fact from a missing category.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        "| Modules | {} |".format(summary["modules"]),
        "| Classes/types | {} |".format(summary["types"]),
        "| Exports | {} |".format(summary["exports"]),
        "| Export rows | {} |".format(summary["export_rows"]),
        "| Runtime facts | {} |".format(summary["runtime_facts"]),
        "| Shader facts | {} |".format(summary["shader_facts"]),
        "| Platforms | {} |".format(summary["platforms"]),
        "| Normalised facts | {} |".format(summary["facts"]),
        "| Conflicts | {} |".format(summary["conflicts"]),
        "| Shim exports | {} |".format(summary["shims"]),
        "| Shims without fact | {} |".format(summary["shims_without_fact"]),
        "",
        "## Categories",
        "",
        "| Category | State | Files | Facts |",
        "| --- | --- | ---: | ---: |",
    ]
    for category in CATEGORIES:
        info = report["categories"][category]
        files = ", ".join(info["files"]) if info["files"] else "-"
        lines.append(
            "| {} | {} | {} | {} |".format(
                category,
                "present" if info["present"] else "missing",
                _md_cell(files),
                info["facts"],
            )
        )
    lines.append("")
    if report["missing"]:
        lines.append(
            "Missing categories: {}.".format(", ".join(report["missing"]))
        )
    else:
        lines.append("All six categories are present.")
    if report["required"]:
        if report["required_missing"]:
            lines.append(
                "Required categories missing: {}.".format(
                    ", ".join(report["required_missing"])
                )
            )
        else:
            lines.append(
                "Required categories present: {}.".format(
                    ", ".join(report["required"])
                )
            )

    lines.extend(["", "## Conflicts", ""])
    if report["conflicts"]:
        lines.extend(
            [
                "| Kind | Module | Name | Values | Sources |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for conflict in report["conflicts"]:
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    conflict["kind"],
                    _md_cell(conflict["module"]),
                    _md_cell(conflict["name"]),
                    _md_cell(_md_values(conflict["values"])),
                    _md_cell(", ".join(conflict["sources"])),
                )
            )
    else:
        lines.append("No conflicts detected.")

    lines.extend(
        [
            "",
            "## Coverage by module",
            "",
            "| Module | Classes/types | Exports | Runtime facts | Shader facts | Platform facts | Facts |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for module in report["modules"]:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                _md_cell(module["module"]),
                module["types"],
                module["exports"],
                module["runtime_facts"],
                module["shader_facts"],
                module["platform_facts"],
                module["facts"],
            )
        )

    lines.extend(["", "## Shims without associated fact", ""])
    if report["inputs"]["defs"] is None:
        lines.append("Shim coverage was not checked; pass --defs or a mapping JSON.")
    elif report["shims"]["without_fact"]:
        lines.extend(
            [
                "| Module | Shim export | Alias | Def |",
                "| --- | --- | --- | --- |",
            ]
        )
        for shim in report["shims"]["without_fact"]:
            lines.append(
                "| {} | {} | {} | {} |".format(
                    _md_cell(shim["module"]),
                    _md_cell(shim["name"]),
                    _md_cell(shim["alias"] or "-"),
                    _md_cell(shim["source"]),
                )
            )
    else:
        lines.append(
            "All {} shim exports match a fact.".format(report["shims"]["exports"])
        )

    lines.extend(
        [
            "",
            "## Limits",
            "",
            "A header fact does not prove the runtime. Layout and ABI rows only describe",
            "the header or compile-time probe that produced them; runtime observations are",
            "separate facts with their own source and confidence. An absent category is",
            "reported as missing and is not treated as a zero-valued fact. Ordinals are",
            "compared only when both sources report one, and RVA differences between",
            "debug, release, import-library and DLL records are not conflicts. Anonymous",
            "MSVC tags are counted as types but are not compared by display name because",
            "several distinct anonymous records can share the same text.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_text(path, text):
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise InputError("cannot write {}: {}".format(path, exc)) from exc


def _parse_required(value):
    if value is None:
        return []
    result = []
    for token in value.split(","):
        category = token.strip().casefold()
        if not category:
            raise InputError("--require contains an empty category")
        if category not in CATEGORIES:
            raise InputError(
                "--require has unknown category {}; expected one of {}".format(
                    category, ", ".join(CATEGORIES)
                )
            )
        if category not in result:
            result.append(category)
    return result


def _run_report(args):
    root = Path(args.root)
    json_path = Path(args.json)
    markdown_path = Path(args.markdown)
    if json_path.resolve() == markdown_path.resolve():
        raise InputError("--json and --markdown must be different paths")
    required = _parse_required(args.require)
    report = build_report(root, args.defs, required)

    _write_text(
        json_path,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _write_text(markdown_path, render_markdown(report))

    summary = report["summary"]
    print(
        "{}: {} facts, {} conflicts, {} shims without fact".format(
            TOOL,
            summary["facts"],
            summary["conflicts"],
            summary["shims_without_fact"],
        )
    )

    if args.check:
        failures = []
        if report["conflicts"]:
            failures.append("{} conflict(s)".format(len(report["conflicts"])))
        if report["required_missing"]:
            failures.append(
                "missing required categories: {}".format(
                    ", ".join(report["required_missing"])
                )
            )
        if failures:
            print("error: " + "; ".join(failures), file=sys.stderr)
            return 1
    return 0


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Merge GDK knowledge facts and report coverage and conflicts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    report = subparsers.add_parser("report", help="write coverage JSON and Markdown")
    report.add_argument("--root", required=True, help="knowledge root directory")
    report.add_argument("--json", required=True, help="coverage JSON output")
    report.add_argument("--markdown", required=True, help="Markdown report output")
    report.add_argument(
        "--defs",
        help="repository src directory, one .def file, or a shim-mapping JSON",
    )
    report.add_argument(
        "--require",
        help="comma-separated categories that must be present (with --check)",
    )
    report.add_argument(
        "--check",
        action="store_true",
        help="exit 1 on conflicts or missing categories listed by --require",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            return _run_report(args)
        parser.error("unknown command")
    except InputError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
