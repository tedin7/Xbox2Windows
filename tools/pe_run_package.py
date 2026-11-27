"""Load an acyclic, source-built x64 Windows PE package through pe_map."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import ctypes as C
from ctypes import wintypes as W
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import pefile

from pe_package_preflight import inspect
from host_commit import read_host_commit, require_commit_budget

# --trap-callable exit codes are TRAP_EXIT_BASE | trap number (1-based, in sorted dll!symbol order).
TRAP_EXIT_BASE = 0xE7A00000
TRAP_LIMIT = 0xFFFF
_TRAP_CALLBACKS = []  # Process lifetime: trap thunks must outlive every guest thread.


def plan(report, *, allow_external=True):
    """Resolve the entire import graph before any guest attachment.

    The default keeps external host imports as unverified candidates. Set
    allow_external=False for a strict closed-package plan: every reachable
    import must terminate in an export bundled in the package. This pure check
    runs before any module is loaded or mapped.
    """
    accepted = {"bundled_export_present"}
    if allow_external:
        accepted.update(("external_unverified", "forwarded_external_unverified"))
    routes = defaultdict(dict)
    needs = defaultdict(set)
    for row in report["imports"]:
        image, name = row["image"], (row["dll"].casefold(), row["symbol"])
        if row["status"] not in accepted:
            raise ValueError(f"{image}: {name[0]}!{name[1]}: {row['status']}")
        chain = row["chain"]
        if not chain:
            raise ValueError(f"{image}: {name[0]}!{name[1]}: empty resolution chain")
        for caller, provider in zip([image, *(step["provider"] for step in chain)], chain):
            if provider.get("package") and caller != provider["provider"]:
                needs[caller].add(provider["provider"])
        terminal = chain[-1]
        if terminal.get("package") and terminal["provider"] == image:
            raise ValueError(f"{image}: self import needs deferred DLL attachment")
        old = routes[image].setdefault(name, terminal)
        if old != terminal:
            raise ValueError(f"{image}: conflicting bindings for {name[0]}!{name[1]}")

    order = []
    pending = set(report["reachable"])
    while pending:
        finished = sorted(image for image in pending if not needs[image] & pending)
        if not finished:
            raise ValueError("bundled dependency cycle: " + ", ".join(sorted(pending)))
        order.extend(finished)
        pending.difference_update(finished)
    if not order:
        raise ValueError("reachable package graph is empty")
    if order[-1] != report["entry"]:
        raise ValueError("entry is a dependency of a bundled image")
    return order, routes


def is_host_create_thread(dll, symbol):
    """A host CreateThread import: kernel32, kernelbase or a processthreads API-set contract."""
    dll = dll.casefold()
    return symbol == "CreateThread" and (dll in ("kernel32.dll", "kernelbase.dll")
                                         or dll.startswith("api-ms-win-core-processthreads-"))


def prototype(lib, name, args, result):
    function = getattr(lib, name)
    function.argtypes, function.restype = args, result
    return function


SEM_FAILCRITICALERRORS = 0x0001


def quiet_loader(kernel, load):
    """LoadLibraryExW with SEM_FAILCRITICALERRORS set on the calling thread for the call only.

    A provider that is not a loadable image (empty, truncated or not a PE) raises a critical error
    inside the loader. Without the flag Windows shows a modal "Bad Image" dialog and the load blocks
    until someone closes it; with it the load just fails, and the provider's symbols are reported
    missing. The thread's previous mode is restored, so the guest runs under the host's setting.
    """
    get_mode = prototype(kernel, "GetThreadErrorMode", [], W.DWORD)
    set_mode = prototype(kernel, "SetThreadErrorMode", [W.DWORD, C.POINTER(W.DWORD)], W.BOOL)

    def call(path, flags):
        previous = W.DWORD()
        changed = set_mode(get_mode() | SEM_FAILCRITICALERRORS, C.byref(previous))
        try:
            return load(path, None, flags)
        finally:
            if changed:
                set_mode(previous.value, None)

    return call


def trap_key(name):
    """Parse a --trap-callable DLL!SYMBOL (or DLL!#ORDINAL) into an import key."""
    dll, sep, symbol = name.partition("!")
    if (not sep or not dll or not symbol or dll in (".", "..")
            or Path(dll).name != dll or "/" in dll or "\\" in dll):
        raise ValueError(f"--trap-callable expects DLL!SYMBOL or DLL!#ORDINAL, got {name!r}")
    if symbol.startswith("#"):
        ordinal = symbol[1:].lstrip("0")
        if not (1 <= len(ordinal) <= 5 and ordinal.isascii() and ordinal.isdigit() and int(ordinal) <= 0xFFFF):
            raise ValueError(f"--trap-callable expects DLL!SYMBOL or DLL!#ORDINAL, got {name!r}")
        symbol = "#" + ordinal
    return dll.casefold(), symbol


def bind_traps(kernel, trap_importers):
    """Bind each listed callable import to a numbered trap for a diagnostic run.

    trap_importers maps (dll, symbol) to the images that import it. A trap never returns
    to its caller: it writes its number and name to stderr, best effort, and terminates
    the process with TRAP_EXIT_BASE | number even when that report fails. No DLL is
    loaded or written, so a trap cannot be mistaken for a provider. Returns
    (addresses, callbacks, labels); keep the callbacks alive for the whole run.
    """
    if len(trap_importers) > TRAP_LIMIT:
        raise ValueError(f"--trap-callable supports at most {TRAP_LIMIT} traps, got {len(trap_importers)}")
    terminate = prototype(kernel, "TerminateProcess", [W.HANDLE, W.UINT], W.BOOL)
    current = prototype(kernel, "GetCurrentProcess", [], W.HANDLE)
    trap_type = C.WINFUNCTYPE(None)
    addresses, callbacks, labels = {}, [], []
    for number, key in enumerate(sorted(trap_importers), 1):
        label = f"{key[0]}!{key[1]}"
        message = f"trap {number}: {', '.join(sorted(trap_importers[key]))}: {label}\n".encode("ascii", "replace")

        def hit(message=message, code=TRAP_EXIT_BASE | number):
            # ctypes swallows an exception raised here and returns to the guest, so the report
            # is best effort and termination happens in finally, whatever the report did.
            try:
                try:
                    os.write(2, message)  # unbuffered, so the line is out before the process ends
                except BaseException:
                    pass
            finally:
                try:
                    terminate(current(), code)
                finally:
                    os._exit(code)

        trap = trap_type(hit)
        callbacks.append(trap)
        addresses[key] = C.cast(trap, C.c_void_p).value
        labels.append(label)
    return addresses, callbacks, labels


def run(root, report, mapper, shims, system, *, preflight_only=False, pin_shims=(), trap_callable=()):
    order, routes = plan(report)
    callback = C.WINFUNCTYPE(C.c_void_p, C.c_char_p, C.c_char_p, C.c_uint32)
    kernel = C.WinDLL("kernel32", use_last_error=True)
    load = quiet_loader(kernel, prototype(kernel, "LoadLibraryExW", [W.LPCWSTR, W.HANDLE, W.DWORD], W.HMODULE))
    release_host = prototype(kernel, "FreeLibrary", [W.HMODULE], W.BOOL)
    find = prototype(kernel, "GetProcAddress", [W.HMODULE, C.c_void_p], C.c_void_p)
    pin_module = prototype(kernel, "GetModuleHandleExW", [W.DWORD, C.c_void_p, C.POINTER(W.HMODULE)], W.BOOL)
    modules, addresses = {}, {}
    if set(pin_shims) - shims.keys():
        raise ValueError("Pinned providers must be named with --shim")
    trap_keys = {trap_key(name) for name in trap_callable}
    if trap_keys and preflight_only:
        raise ValueError("--trap-callable applies to execution mode, not --preflight-only")
    trap_labels = []
    with ExitStack() as stack:
        missing, missing_keys = [], defaultdict(set)
        missing_entry_keys = []
        for image in order:
            for terminal in routes[image].values():
                if terminal.get("package"):
                    continue
                dll, symbol = terminal["dll"].casefold(), terminal["symbol"]
                key = (dll, symbol)
                if key in addresses:
                    if not addresses[key]:
                        missing_keys[key].add(image)
                    continue
                if dll not in modules:
                    module = (load(str(shims[dll]), 0x900) if dll in shims else
                              load(dll, 0x800) if dll in system else None)
                    modules[dll] = module
                    if module:
                        stack.callback(release_host, module)
                module = modules[dll]
                needle = (int(symbol[1:]) if symbol.startswith("#")
                          else C.cast(C.c_char_p(symbol.encode("ascii")), C.c_void_p))
                addresses[key] = find(module, needle) if module else None
                if not addresses[key]:
                    missing.append(f"{image}: {dll}!{symbol}")
                    missing_keys[key].add(image)
                    missing_entry_keys.append(key)
        if trap_keys:
            # An import table entry does not say whether the caller calls the symbol or reads it as
            # data, so only names the caller lists as callable get a trap; any other unresolved
            # external is refused as before.
            stale = sorted(trap_keys - missing_keys.keys())
            if stale:
                raise ValueError(f"--trap-callable {stale[0][0]}!{stale[0][1]} is not an unresolved external")
            unlisted = [entry for entry, key in zip(missing, missing_entry_keys) if key not in trap_keys]
            if unlisted:
                raise ValueError(f"{len(unlisted)} unresolved external symbol(s) not listed with "
                                 f"--trap-callable, first {unlisted[0]}")
            trap_addresses, trap_callbacks, trap_labels = bind_traps(
                kernel, {key: missing_keys[key] for key in trap_keys})
            # A guest worker may still reach a trap after the entry returns.
            _TRAP_CALLBACKS.extend(trap_callbacks)
            addresses.update(trap_addresses)
            sys.stderr.write(f"trap-callable: {len(trap_labels)} listed unresolved external(s) bound to "
                             "terminating traps; diagnostic only, not providers\n")
            for number, label in enumerate(trap_labels, 1):
                sys.stderr.write(f"  trap {number} (exit {TRAP_EXIT_BASE | number:#x}): {label}\n")
            sys.stderr.flush()
        elif missing and not preflight_only:
            raise ValueError(f"{len(missing)} unresolved external symbol(s), first {missing[0]}")
        if preflight_only:
            return {"image_order": order, "external_bindings": len(addresses) - len(missing),
                    "missing": missing, "entry_executed": False}

        # Recheck the reachable files before the first guest attach.
        for image in order:
            with (root / image).open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != report["images"][image]["sha256"]:
                raise ValueError(f"{image}: file changed after inventory")

        lib = C.WinDLL(str(mapper), winmode=0x900)
        ordinal_routes = any(terminal.get("package") and terminal["symbol"].startswith("#")
                             for bindings in routes.values() for terminal in bindings.values())
        export_ordinal = None
        if ordinal_routes:
            try:
                export_ordinal = prototype(lib, "pe_map_export_ordinal", [C.c_void_p, C.c_uint32], C.c_void_p)
            except AttributeError as exc:
                raise ValueError("bundled ordinal imports require a mapper with pe_map_export_ordinal") from exc
        for name in sorted(pin_shims):
            module = modules.get(name)
            if not module:
                raise ValueError(f"{name}: pinned provider is not in the resolved import plan")
        # Provider workers may outlive guest return. Balance LoadLibrary normally
        # but retain these modules until the isolated process itself exits.
        pin_from_address = 0x1 | 0x4  # GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
        for name in sorted(pin_shims):
            handle = W.HMODULE()
            if not pin_module(pin_from_address, modules[name], C.byref(handle)):
                raise ValueError(f"{name}: cannot pin provider ({C.get_last_error()})")
            release_host(handle)
        map_image = prototype(lib, "pe_map_image",
                              [W.LPCWSTR, callback, C.POINTER(C.c_uint32), C.c_char_p, C.c_size_t], C.c_void_p)
        export = prototype(lib, "pe_map_export", [C.c_void_p, C.c_char_p], C.c_void_p)
        unmap = prototype(lib, "pe_map_release", [C.c_void_p], C.c_int)
        # The entry is the package's main executable, as under a Windows launch: pe_map_run_main gives
        # its TLS block slot 0 on the participating threads (code built with /GA reads slot 0 directly),
        # and its host CreateThread imports go to pe_map_main_create_thread so its workers participate.
        try:
            start = prototype(lib, "pe_map_run_main", [C.c_void_p, C.POINTER(C.c_uint32)], C.c_int)
            enroll = C.cast(lib.pe_map_main_create_thread, C.c_void_p).value
        except AttributeError as exc:
            raise ValueError("the entry EXE requires a mapper with pe_map_run_main") from exc
        mapped = {}
        for image in order:
            imports = {}
            for key, terminal in routes[image].items():
                symbol = terminal["symbol"]
                if terminal.get("package"):
                    provider = terminal["provider"]
                    imports[key] = (export_ordinal(mapped[provider], int(symbol[1:]))
                                    if symbol.startswith("#") else
                                    export(mapped[provider], symbol.encode("ascii")))
                    if not imports[key]:
                        raise ValueError(f"{image}: mapped export {provider}!{symbol} unavailable")
                elif (image == report["entry"]
                      and terminal["dll"].casefold() not in shims
                      and is_host_create_thread(terminal["dll"], symbol)):
                    imports[key] = enroll
                else:
                    imports[key] = addresses[terminal["dll"].casefold(), symbol]

            @callback
            def resolve(dll, name, ordinal):
                try:
                    symbol = name.decode("ascii") if name else f"#{ordinal}"
                    return imports.get((dll.decode("ascii").casefold(), symbol))
                except UnicodeDecodeError:
                    return None

            status, detail = C.c_uint32(), C.create_string_buffer(512)
            mapped[image] = map_image(str(root / image), resolve, C.byref(status), detail, len(detail))
            if not mapped[image]:
                raise ValueError(f"{image}: mapper refused ({status.value}): {detail.value.decode(errors='replace')}")
            stack.callback(unmap, mapped[image])
        code = C.c_uint32()
        if not start(mapped[report["entry"]], C.byref(code)):
            raise ValueError("mapper refused the EXE start")
        result = {"image_order": order, "entry_executed": True, "exit_code": code.value}
        if trap_keys:
            # Reaching this point means no trap was called.
            result["unhit_traps"] = trap_labels
        return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapper", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("entry")
    parser.add_argument("--shim", action="append", nargs=2, default=[], metavar=("DLL", "PATH"))
    parser.add_argument("--pin-shim", action="append", default=[], metavar="DLL",
                        help="keep a named external shim loaded until this process exits")
    parser.add_argument("--system", action="append", default=[], metavar="DLL")
    parser.add_argument("--min-commit-mib", type=int, metavar="MIB",
                        help="optional Windows system commit-headroom check before loading any DLL; "
                             "0 reports headroom without imposing a minimum")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--trap-callable", action="append", default=[], metavar="DLL!SYMBOL",
                      help="diagnostic: bind this unresolved external, which the caller must call rather than "
                           "read as data, to a trap that names it on stderr and terminates the process with "
                           f"{TRAP_EXIT_BASE:#x} | trap number; any unlisted unresolved external is still refused")
    options = parser.parse_args(argv)
    try:
        # Perform the opt-in check before parsing the package or loading providers.
        # A successful check does not reserve memory and cannot promise startup.
        commit_report = None
        if options.min_commit_mib is not None:
            if not 0 <= options.min_commit_mib <= 1 << 20:
                raise ValueError("--min-commit-mib must be between 0 and 1048576")
            commit_report = require_commit_budget(options.min_commit_mib, read_host_commit())
        root = options.package.resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(f"package path is not a directory: {root}")
        mapper = options.mapper
        if mapper.is_dir():
            raise IsADirectoryError(f"mapper path is a directory: {mapper}")
        shims = {}
        for name, raw_path in options.shim:
            shim_path = Path(raw_path).resolve(strict=True)
            if shim_path.is_dir():
                raise IsADirectoryError(f"--shim {name} path is a directory: {shim_path}")
            if not shim_path.is_file():
                raise ValueError(f"--shim {name} path is not a regular file: {shim_path}")
            shims[name.casefold()] = shim_path
        system = {name.casefold() for name in options.system}
        if any(not name or name in (".", "..") or Path(name).name != name or "/" in name or "\\" in name
               for name in system):
            raise ValueError("System providers must be DLL basenames")
        report = inspect(root, options.entry)
        result = run(root, report, mapper.resolve(), shims, system,
                     preflight_only=options.preflight_only,
                     pin_shims={name.casefold() for name in options.pin_shim},
                     trap_callable=options.trap_callable)
        if commit_report is not None:
            result = {**result, "host_commit": commit_report}
    except (ValueError, OSError, pefile.PEFormatError, struct.error) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
