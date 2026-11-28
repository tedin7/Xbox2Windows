"""Statically bind the inspected loader's create path to a host-owned provider table."""
import argparse
import os
from pathlib import Path
import tempfile

import pefile

from ffx_version_query import INTEGER_HEADER, LOADER, checked_image, export_body, lower


TYPES = """
struct Allocator {
    void* user;
    void* (*allocate)(void*, uint64_t);
    void (*deallocate)(void*, void*);
};
struct Provider {
    uint32_t (*create)(void**, const void*, const Allocator*);
    uint32_t (*destroy)(void**, const Allocator*);
};
struct Host { const Provider* table; void* (*zero_alloc)(size_t, size_t); };
"""


def create_source(body, globals_, calls, exits, initial):
    # The guest owns the create output slot until the wrapper is published.
    bindings = {
        "resolve": "/* Provider availability is bound by the host. */",
        "create": """{ provider_slot = rcx;
        void* inner = load<void*>(provider_slot);
        rax = host->table->create(&inner, reinterpret_cast<const void*>(rdx),
                                  reinterpret_cast<const Allocator*>(r8));
        store<void*>(provider_slot, inner); }""",
        "allocate": "rax = reinterpret_cast<word>(allocator->allocate(reinterpret_cast<void*>(rcx), rdx));",
        "calloc": "rax = reinterpret_cast<word>(host->zero_alloc(rcx, rdx));",
    }
    # The inspected wrapper store assumes successful allocation. At the host boundary,
    # unwind the already-created inner context if the wrapper cannot be allocated.
    on_failure = """
        if (!rax) {
            void* inner = load<void*>(provider_slot);
            host->table->destroy(&inner, allocator);
            return 5;
        }"""
    bound = {ip: bindings[name] + (on_failure if name in ("allocate", "calloc") else "")
             for ip, name in calls.items()}
    code = lower(body, initial, globals_, bound, exits)
    return INTEGER_HEADER + TYPES + """
extern "C" __declspec(dllexport) uint32_t create_context(void** output, const void* desc,
                                                          const Allocator* allocator, const Host* host) {
    if (!host || (allocator && (!allocator->allocate || !allocator->deallocate))
        || (!allocator && !host->zero_alloc)) return 6;
    const word available = host->table && host->table->create && host->table->destroy;
    const word absent = 0;
    word provider_slot = 0;
""" + code + "\n}\n"


def generate(path):
    with checked_image(path, LOADER) as pe:
        body = export_body(pe, "ffxCreateContext", [(0x117f, 0x11a2), (0x11a2, 0x125e),
                                                     (0x1273, 0x127a)])
    globals_ = {0x4088: "reinterpret_cast<word>(&available)", 0x4090: "reinterpret_cast<word>(host->table)",
                0x40c0: "reinterpret_cast<word>(&absent)", 0x40c8: "reinterpret_cast<word>(&absent)"}
    calls = {0x11a7: "resolve", 0x121f: "create", 0x1232: "allocate", 0x123c: "calloc"}
    return create_source(body, globals_, calls, {0x125e: "return static_cast<uint32_t>(rax);", 0x127a: "return 6;"},
                         ["rcx = reinterpret_cast<word>(output); rdx = reinterpret_cast<word>(desc);",
                          "r8 = reinterpret_cast<word>(allocator);"])


def _validate_paths(loader: Path, output: Path, parser: argparse.ArgumentParser) -> None:
    if loader.is_dir():
        raise IsADirectoryError(f"loader path is a directory: {loader}")
    if not loader.is_file():
        raise FileNotFoundError(f"loader file not found: {loader}")
    if output.suffix.lower() != ".cpp":
        parser.error("choose a separate .cpp output")
    if output.is_symlink():
        raise ValueError(f"output path must not be a symlink: {output}")
    if output.is_dir():
        raise IsADirectoryError(f"output path is a directory: {output}")
    for parent in output.parents:
        if parent.is_file() or ((parent.exists() or parent.is_symlink()) and not parent.is_dir()):
            raise NotADirectoryError(f"output parent is not a directory: {parent}")
    if output.resolve() == loader.resolve() or (output.exists() and loader.samefile(output)):
        parser.error("choose a separate .cpp output")


def _write_atomic(path: Path, content: str) -> None:
    """Write ASCII content to a sibling temp file and atomically replace path, leaving 0 temp files on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("loader", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        _validate_paths(args.loader, args.output, parser)
        _write_atomic(args.output, generate(args.loader))
    except (OSError, ValueError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
