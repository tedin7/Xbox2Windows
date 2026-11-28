# GDK knowledge coverage report

tools/gdk_knowledge_report.py merges the JSON knowledge base under
D:\gdk-oracle\knowledge and writes a deterministic coverage JSON and a
Markdown summary. It uses only the Python standard library and never reads a
GDK header, library or binary.

## Inputs

All categories are optional. A missing directory or file is reported as a
missing category; it is not treated as an empty fact.

| Path | Accepted shape |
| --- | --- |
| layouts/*.json | classes array with name, size and members, or a class array |
| abi/*.json | facts array with kind, name and value, or a fact array |
| exports/*.json | exports array with name, ordinal and rva, module groups, or a module map |
| exports-all.json | the same export rows, including the file and kind group form |
| behavior/*.json | observations array, or an observation array |
| platform/*.json | define, include and library lists, plus a flag list or object |
| shaders/corpus.json or shaders/corpus/corpus.json | cases array, or a case array |

Every fact is normalised to module, kind, name, value, source, confidence and
targets. Layout classes become layout facts whose value contains size and
members; ABI sizeof and struct facts keep their kind. Exports carry their
ordinal and RVA. Runtime observations become behavior facts, and the
observation kind is retained inside the value when it adds information.
Modules are merged case-insensitively.

A top-level source of msvc-layout, compiled-probe, dumpbin, runtime-oracle or
header-msvc-layout is treated as a method: it becomes the confidence and the
source stays the JSON path. Other source values are kept as given.

## Conflicts

The report emits stable value, size, ordinal, enum and type-size entries. Each
entry names the module, the fact, every distinct value with its source, and a
short message.

- size compares all size-bearing facts for one module and type, including a
  layout member list and an ABI sizeof fact.
- type-size compares a named type across modules and reports incompatible
  sizes. Equal sizes in two modules are not a conflict.
- ordinal compares only ordinals that both sources report. A missing ordinal
  in an import library and a present ordinal in a DLL do not conflict.
- enum compares enum facts with the same module and name.
- value covers every other fact with the same module, kind and name.
- RVAs are retained in the facts but are not compared: Debug, Release, import
  library and DLL records legitimately differ.

MSVC anonymous tags include the text unnamed-tag in their display name. Several
distinct anonymous unions or structs can share that text, so they are kept as
facts and counted as types but excluded from conflict comparisons. Their
display name is not a stable identity.

## Coverage and shims

The summary counts modules, unique classes/types, unique exports and raw export
rows, runtime facts, shader facts, platforms and all normalised facts. A
per-module table gives the same breakdown.

--defs accepts the repository src directory, one .def file, or a JSON mapping.
The parser reads LIBRARY, export names, aliases and ordinals, and ignores
internal x2w_* and X2W* helpers. A shim export is associated when its name or
alias target matches any fact name. The JSON and Markdown list the remaining
exports.

## Commands

    python -B tests/gdk_knowledge_report_check.py

    $root = 'D:\gdk-oracle\knowledge'
    python -B tools/gdk_knowledge_report.py report --root $root --json "$root\coverage.json" --markdown "$root\report.md" --defs src

Add --check to exit 1 on conflicts. Add --require layouts,abi,exports to also
exit 1 when one of those categories is missing. Bad JSON, a bad schema, an
unknown required category or an unusable path exits 2. Without --check,
missing categories and conflicts are only reported.

## Measured run

On 2026-11-28, the local knowledge root produced 45 modules, 11,145 unique
types, 4,714 unique exports from 8,035 export rows, 12 runtime facts, 6 shader
cases and 3 platform profiles. The 9,677 ABI sizeof facts matched their layout
sizes, so the conflict list was empty. All six categories were present. All 70
GDK-facing shim exports matched an export fact.

## Limits

A header fact does not prove the runtime. A layout or ABI row only reports what
the header or compile probe describes. A runtime observation is separate
evidence. An export name proves neither a signature nor a working
implementation, and a shim name match does not prove matching behaviour. The
report intentionally does not compare RVAs, infer missing categories, or turn
an absent oracle into a pass. The generated coverage JSON can be large because
it retains every normalised fact; keep it outside the repository with the rest
of the knowledge base.
