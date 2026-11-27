"""Opt-in content-addressed cache for DXC compiles in the Python shader tests.

Active only when the environment variable X2W_DXC_CACHE names a directory.
Unset, run_dxc is a plain subprocess.run. Anything the cache cannot reason
about (includes, -I, response files, unknown -F outputs, extra run options)
bypasses it, so a cached result is never reused for a different compile.
"""
import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ENV_VAR = "X2W_DXC_CACHE"
OUTPUT_FLAGS = ("-Fo", "-Fc", "-Fh", "-Fre", "-Fd")
SUPPORTED_KWARGS = {"check", "capture_output", "text", "timeout"}
INCLUDE_RE = re.compile(rb"#\s*(include|import)\b|@import")
KEY_VERSION = "2"


def _plan(compiler, args):
    """Return (key_args, input_path, outputs) or None when the compile is not cacheable."""
    args = [str(a) for a in args]
    if not args:
        return None
    source = args[-1]
    if source.startswith("-") or not os.path.isfile(source):
        return None
    key_args = []
    outputs = {}
    for index, arg in enumerate(args[:-1]):
        if arg.startswith("@") or arg.startswith("-I") or arg.startswith("-Fi"):
            return None
        if arg.startswith("-F") and arg not in OUTPUT_FLAGS:
            return None
        if index > 0 and args[index - 1] in OUTPUT_FLAGS:
            continue  # output path value, handled with its flag
        if arg in OUTPUT_FLAGS:
            if index + 1 >= len(args) - 1:
                return None
            path = args[index + 1]
            if path.endswith(("/", "\\")) or os.path.isdir(path) or path in outputs.values() or path == source:
                return None
            if not os.path.isdir(os.path.dirname(os.path.abspath(path))):
                return None
            outputs[arg] = path
            key_args.append(arg)
            key_args.append("<out:%s>" % arg)
        elif os.path.exists(arg):
            return None  # unknown file-like argument
        else:
            key_args.append(arg)
    if not outputs:
        return None
    return key_args + ["<in>"], source, outputs


def _scrub(data, mapping):
    for text, token in mapping:
        data = data.replace(text.encode(), token.encode())
    return data


def _restore(data, mapping):
    for text, token in mapping:
        data = data.replace(token.encode(), text.encode())
    return data


def _store(path, payload):
    handle, temp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(payload)
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise


def _write_file(path, data):
    """Write data to path atomically, refusing symlink or directory targets."""
    target = os.fspath(path)
    stripped = os.path.normpath(target)
    if os.path.islink(target) or os.path.islink(stripped):
        raise OSError(errno.ELOOP, "refusing symlink target", target)
    if target.endswith(("/", "\\")) or os.path.isdir(target):
        raise IsADirectoryError(errno.EISDIR, "refusing directory target", target)
    parent = os.path.dirname(os.path.abspath(target))
    handle, temp = tempfile.mkstemp(dir=parent, prefix=".tmp-", suffix=".tmp")
    closed = False
    try:
        with os.fdopen(handle, "wb") as out:
            closed = True
            out.write(data)
        os.replace(temp, target)
    except BaseException:
        if not closed:
            os.close(handle)
        if os.path.exists(temp):
            os.unlink(temp)
        raise


def _valid_cache_dir(cache_dir):
    """Return True when cache_dir is usable and neither symlink- nor file-backed."""
    if not cache_dir:
        return False
    target = os.path.normpath(os.fspath(cache_dir))
    if os.path.islink(target):
        return False
    if os.path.lexists(target):
        return os.path.isdir(target)
    probe = os.path.dirname(os.path.abspath(target))
    while probe:
        if os.path.lexists(probe):
            return os.path.isdir(probe)
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return True


def _emit(result_bytes, text, stream):
    stream.write(result_bytes.decode(errors="replace") if text else result_bytes)  # pragma: no cover
    stream.flush()


def _passthrough(options, check):
    """Keyword arguments exactly as the caller gave them (check only when given)."""
    return dict(options, check=check) if check is not None else dict(options)


def _resolve(compiler):
    """Absolute path of the compiler (a file, else found on PATH), or None when it is neither."""
    name = os.fspath(compiler)
    if os.path.isfile(name):
        return os.path.abspath(name)
    found = shutil.which(name)
    return os.path.abspath(found) if found else None


def _is_wrapper(resolved):
    """True for a shell/script launcher that delegates to a compiler we cannot fingerprint (Wine)."""
    try:
        with open(resolved, "rb") as handle:
            head = handle.read(4096)
    except OSError:
        return True
    return head.startswith(b"#!") and b"wine" in head.lower()


COMPILER_LIBRARIES = ("dxcompiler.dll", "dxil.dll", "libdxcompiler.so", "libdxil.so")


def _libraries(resolved):
    """Name, size and mtime of the DXC libraries next to the compiler (missing ones are skipped)."""
    folder = os.path.dirname(resolved)
    found = []
    for name in COMPILER_LIBRARIES:
        try:
            info = os.stat(os.path.join(folder, name))
        except OSError:
            continue
        found.append([name, info.st_size, info.st_mtime_ns])
    return found


def run_dxc(compiler, args, **options):
    """Mirror subprocess.run([compiler, *args]) with an optional on-disk cache."""
    command = [compiler, *args]
    check = options.pop("check", None)
    capture_output = options.get("capture_output", False)
    text = options.get("text", False)
    timeout = options.get("timeout")
    extra = {k: v for k, v in options.items() if k not in SUPPORTED_KWARGS}
    cache_dir = os.environ.get(ENV_VAR)
    plan = None
    resolved = None
    if _valid_cache_dir(cache_dir) and not extra:
        resolved = _resolve(compiler)
        if resolved is not None and not _is_wrapper(resolved):
            plan = _plan(compiler, args)
    if plan is None:
        return subprocess.run(command, **_passthrough(options, check))
    key_args, source, outputs = plan
    with open(source, "rb") as handle:
        source_bytes = handle.read()
    if INCLUDE_RE.search(source_bytes):
        return subprocess.run(command, **_passthrough(options, check))
    try:
        info = os.stat(resolved)
    except OSError:
        return subprocess.run(command, **_passthrough(options, check))
    digest = hashlib.sha256()
    digest.update(json.dumps([KEY_VERSION, resolved, info.st_size, info.st_mtime_ns,
                              _libraries(resolved), key_args]).encode())
    digest.update(b"\0")
    digest.update(source_bytes)
    entry = os.path.join(cache_dir, digest.hexdigest() + ".json")
    mapping = [(source, "<in>")] + [(path, "<out:%s>" % flag) for flag, path in outputs.items()]
    mapping.sort(key=lambda item: -len(item[0]))
    cached = None
    try:
        with open(entry, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
    except (OSError, ValueError):
        cached = None
    if cached is not None:
        try:
            files = {flag: bytes.fromhex(cached["files"][flag]) for flag in cached["files"]}
            stdout = _restore(bytes.fromhex(cached["stdout"]), mapping)
            stderr = _restore(bytes.fromhex(cached["stderr"]), mapping)
            returncode = int(cached["returncode"])
            if returncode != 0:
                raise ValueError("failures are never replayed")
        except (KeyError, ValueError, TypeError):
            cached = None
    if cached is None:
        done = subprocess.run(command, capture_output=True, timeout=timeout)
        stdout, stderr, returncode = done.stdout, done.stderr, done.returncode
        files = {}
        for flag, path in outputs.items():
            if os.path.isfile(path):
                with open(path, "rb") as handle:
                    files[flag] = handle.read()
        # Only successes are cached: a failure may be transient (I/O, resources, bad output directory).
        if returncode == 0:
            os.makedirs(cache_dir, exist_ok=True)
            record = {"returncode": returncode, "stdout": _scrub(stdout, mapping).hex(),
                      "stderr": _scrub(stderr, mapping).hex(),
                      "files": {flag: data.hex() for flag, data in files.items()}}
            _store(entry, json.dumps(record).encode())
    elif returncode == 0:
        for flag, data in files.items():
            _write_file(outputs[flag], data)
    if capture_output:
        out, err = stdout, stderr
        if text:
            out, err = stdout.decode(errors="replace"), stderr.decode(errors="replace")
    else:
        out = err = None
        sys.stdout.flush()
        os.write(1, stdout) if stdout else None
        os.write(2, stderr) if stderr else None
    result = subprocess.CompletedProcess(command, returncode, out, err)
    if check:
        result.check_returncode()
    return result


class _SubprocessProxy:
    """Module-like wrapper: subprocess.run goes through run_dxc for DXC-looking compiles only."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def run(self, command, **kwargs):
        if (isinstance(command, (list, tuple)) and len(command) > 3 and "-T" in command
                and "-Fo" in command and not str(command[0]).startswith("-")):
            return run_dxc(command[0], list(command[1:]), **kwargs)
        return self._real.run(command, **kwargs)


def with_dxc_cache(subprocess_module):
    """Return subprocess_module with .run routed through the cache for DXC compiles."""
    return _SubprocessProxy(subprocess_module)
