#!/usr/bin/env python3
"""Static checks for the XGameRuntime thunk table and its module definition."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "xgameruntime_shim.cpp"
DEFINITION = ROOT / "src" / "xgameruntime_shim.def"

REQUIRED_COMPONENTS = {
    "XUser",
    "XGameSave",
    "XPersistentLocalStorage",
    "XPackage",
    "XSystem",
    "XGame",
    "XAsync",
    "XTaskQueue",
}
REQUIRED_IMPLEMENTED = {
    "XUser",
    "XGameSave",
    "XPersistentLocalStorage",
    "XPackage",
    "XSystem",
    "XGame",
}
REQUIRED_IMPLEMENTED_FUNCTIONS = {
    "XPersistentLocalStorageGetPathSize",
    "XPersistentLocalStorageGetPath",
    "XPersistentLocalStorageGetSpaceInfo",
    "XPackageMount",
    "XPackageGetMountPathSize",
    "XPackageGetMountPath",
    "XPackageCloseMountHandle",
    "XSystemGetConsoleId",
    "XSystemGetDeviceType",
    "XGameGetXboxTitleId",
    "XUserAddAsync",
    "XUserAddResult",
    "XGameSaveInitializeProviderAsync",
    "XGameSaveInitializeProviderResult",
    "XAsyncGetStatus",
    "XAsyncGetResultSize",
    "XTaskQueueCreate",
    "XTaskQueueDispatch",
    "XTaskQueueCloseHandle",
    "XPackageMountWithUiAsync",
    "XPackageMountWithUiResult",
}
REQUIRED_STUBS = set()
STATES = {"stub", "implemented_hr", "implemented_void", "implemented_value"}

ROW_RE = re.compile(
    r"^X\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*,\s*"
    r"([A-Za-z_]\w*)\s*,\s*(\([^)]*\))\s*,\s*([A-Za-z_]\w*)\s*,\s*"
    r"(\([^)]*\))\s*,\s*([A-Za-z_]\w*)\s*\)\s*\\?$"
)


def parse_table(source):
    lines = source.splitlines()
    try:
        start = next(i for i, line in enumerate(lines)
                     if "#define XGAMERUNTIME_THUNK_TABLE(X)" in line)
    except StopIteration as error:
        raise AssertionError("XGAMERUNTIME_THUNK_TABLE(X) is missing") from error

    rows = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped:
            break
        match = ROW_RE.match(stripped)
        if match:
            name, component, return_type, parameters, state, args, implementation = match.groups()
            if not parameters.startswith("(") or not parameters.endswith(")") or \
                    parameters.count("(") != parameters.count(")"):
                raise AssertionError(f"malformed signature for {name}: {parameters}")
            if not args.startswith("(") or not args.endswith(")"):
                raise AssertionError(f"malformed implementation arguments for {name}: {args}")
            rows.append((name, component, return_type, parameters, state, args, implementation))
        if not line.rstrip().endswith("\\"):
            break
    if not rows:
        raise AssertionError("XGAMERUNTIME_THUNK_TABLE(X) has no rows")
    return rows


def parse_definition(definition):
    library = None
    exports = []
    in_exports = False
    for raw in definition.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.upper() == "EXPORTS":
            in_exports = True
            continue
        if line.lower().startswith("library "):
            library = line.split(None, 1)[1]
            continue
        if in_exports:
            exports.append(line)
    return library, exports


def validate(source, rows, library, exports):
    errors = []
    names = [row[0] for row in rows]
    components = {row[1] for row in rows}

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        errors.append(f"duplicate export names: {', '.join(duplicates)}")
    if len(names) != len(set(names)):
        errors.append("table contains a duplicate name")
    missing = sorted(REQUIRED_COMPONENTS - components)
    if missing:
        errors.append(f"missing required classes: {', '.join(missing)}")
    for component in sorted(REQUIRED_IMPLEMENTED):
        if not any(row[1] == component and row[4] != "stub" for row in rows):
            errors.append(f"missing implemented contract for {component}")
    for name in sorted(REQUIRED_IMPLEMENTED_FUNCTIONS):
        matches = [row for row in rows if row[0] == name]
        if len(matches) != 1 or matches[0][4] == "stub":
            errors.append(f"{name} must be one implemented table row")
    for name in sorted(REQUIRED_STUBS):
        matches = [row for row in rows if row[0] == name]
        if len(matches) != 1 or matches[0][4] != "stub":
            errors.append(f"{name} must remain an explicit stub")
    for name, _, return_type, _, state, args, implementation in rows:
        if state not in STATES:
            errors.append(f"unknown state for {name}: {state}")
        if state == "stub":
            if return_type != "HRESULT" or args != "()" or implementation != "stub":
                errors.append(f"stub row has inconsistent implementation: {name}")
        elif implementation == "stub":
            errors.append(f"implemented row has no implementation function: {name}")
        elif state == "implemented_hr" and return_type != "HRESULT":
            errors.append(f"implemented_hr row does not return HRESULT: {name}")
        elif state == "implemented_void" and return_type != "void":
            errors.append(f"implemented_void row does not return void: {name}")
        elif state == "implemented_value" and return_type == "void":
            errors.append(f"implemented_value row returns void: {name}")

    if library != "xgameruntime":
        errors.append(f"LIBRARY is {library!r}, expected 'xgameruntime'")
    if any("@" in export for export in exports):
        errors.append("the definition declares an export ordinal")
    if "XGAMERUNTIME_THUNK_TABLE(XGAMERUNTIME_DEFINE_THUNK)" not in source:
        errors.append("the thunk table does not generate the exported functions")
    if "XGAMERUNTIME_THUNK_TABLE(XGAMERUNTIME_TABLE_ROW)" not in source:
        errors.append("the thunk table does not populate the inspectable metadata")

    export_names = [export.split("=", 1)[0].split()[0] for export in exports]
    export_duplicates = sorted({name for name in export_names
                                if export_names.count(name) > 1})
    if export_duplicates:
        errors.append(f"duplicate definition exports: {', '.join(export_duplicates)}")
    if export_names and set(export_names) != set(names):
        errors.append("definition exports do not match the thunk table")
    return errors


def main():
    source = SOURCE.read_text(encoding="utf-8")
    rows = parse_table(source)
    library, exports = parse_definition(DEFINITION.read_text(encoding="utf-8"))
    errors = validate(source, rows, library, exports)

    duplicate_control = validate(source, rows + [rows[0]], library, exports)
    if not any("duplicate" in error for error in duplicate_control):
        errors.append("duplicate-name negative control was not detected")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    components = ",".join(sorted({row[1] for row in rows}))
    states = ",".join(f"{state}={sum(row[4] == state for row in rows)}"
                      for state in sorted(STATES))
    print(f"xgameruntime table: exports={len(rows)} classes={components} "
          f"states={states} duplicate_control=detected PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
