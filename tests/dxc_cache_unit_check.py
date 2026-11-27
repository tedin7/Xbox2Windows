"""Unit and temp-copy mutation suite for tools/dxc_cache.py."""
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
DXC_CACHE_PATH = TOOLS_DIR / "dxc_cache.py"

FAKE_DXC = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
fo = args[args.index("-Fo") + 1]
with open(fo + ".calls", "a", encoding="utf-8") as handle:
    handle.write("x")
src = open(args[-1], "rb").read()
if b"BROKEN" in src:
    sys.stderr.write("error in " + args[-1] + "\\n")
    sys.exit(3)
with open(fo, "wb") as handle:
    handle.write(b"FO:" + src + b":" + fo.encode())
if "-Fc" in args:
    fc = args[args.index("-Fc") + 1]
    with open(fc, "wb") as handle:
        handle.write(b"FC:" + src + b":" + fc.encode())
    sys.stdout.write("asm " + fc + "\\n")
if "-Fd" in args:
    fd = args[args.index("-Fd") + 1]
    with open(fd, "wb") as handle:
        handle.write(b"FD:" + src)
    sys.stdout.write("pdb " + fd + "\\n")
sys.stdout.write("compiled " + args[-1] + " -> " + fo + "\\n")
"""


def _load_module(path, name="dxc_cache_sut"):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_rig(root):
    root = Path(root)
    if os.name == "nt":
        impl = root / "fake_dxc_impl.py"
        impl.write_text(FAKE_DXC, encoding="utf-8")
        compiler = root / "fake_dxc.cmd"
        compiler.write_text(f'@"{sys.executable}" "{impl}" %*\n', encoding="utf-8")
    else:
        compiler = root / "fake_dxc"
        compiler.write_text(FAKE_DXC, encoding="utf-8")
        os.chmod(compiler, os.stat(compiler).st_mode | stat.S_IEXEC)
    return str(compiler)


def _calls(root):
    return sum(len(p.read_text(encoding="utf-8")) for p in Path(root).glob("*.calls"))


def _tmp_files(directory):
    return [p for p in Path(directory).iterdir() if p.name.startswith(".tmp-") or p.name.endswith(".tmp")]


def _make_symlink(target, link_path):
    try:
        os.symlink(target, link_path)
        return True
    except (OSError, NotImplementedError):
        return False


def _test_atomic_restore_and_zero_temps(mod, root, compiler):
    cache_dir = Path(root) / "cache_atomic"
    src = Path(root) / "shader.hlsl"
    src.write_bytes(b"float4 main() : SV_Target { return 1; }")
    out1 = Path(root) / "out1.cso"
    out2 = Path(root) / "out2.cso"
    out2.write_bytes(b"STALE_OUT2")
    with mock.patch.dict(os.environ, {mod.ENV_VAR: str(cache_dir)}):
        res1 = mod.run_dxc(compiler, ["-T", "ps_6_0", "-Fo", str(out1), str(src)],
                           capture_output=True, text=True, check=True)
        res2 = mod.run_dxc(compiler, ["-T", "ps_6_0", "-Fo", str(out2), str(src)],
                           capture_output=True, text=True, check=True)
    assert res1.returncode == 0 and res2.returncode == 0
    assert _calls(root) == 1, f"expected 1 real compile, got {_calls(root)}"
    assert out2.read_bytes().startswith(b"FO:")
    assert b"STALE_OUT2" not in out2.read_bytes()
    assert str(out2) in res2.stdout
    assert not _tmp_files(root), f"leftover temp files in output dir: {_tmp_files(root)}"
    assert not _tmp_files(cache_dir), f"leftover temp files in cache dir: {_tmp_files(cache_dir)}"


def _test_replace_fail_preserves_output_and_cleans_temps(mod, root, compiler):
    cache_dir = Path(root) / "cache_replace_fail"
    src = Path(root) / "replace.hlsl"
    src.write_bytes(b"void main(){}")
    warm = Path(root) / "warm.cso"
    target = Path(root) / "preserve.cso"
    target.write_bytes(b"ORIGINAL_BYTES")

    # Direct _write_file failure via os.replace.
    with mock.patch.object(mod.os, "replace", side_effect=OSError("boom")):
        try:
            mod._write_file(str(target), b"NEW_BYTES")
        except OSError:
            pass
        else:
            raise AssertionError("_write_file did not propagate os.replace OSError")
    assert target.read_bytes() == b"ORIGINAL_BYTES", "prior output corrupted on os.replace failure"
    assert not _tmp_files(root), f"leftover temp files after os.replace failure: {_tmp_files(root)}"

    # Direct _write_file failure via write TypeError.
    try:
        mod._write_file(str(target), "not-bytes")
    except TypeError:
        pass
    else:
        raise AssertionError("_write_file did not reject non-bytes write")
    assert target.read_bytes() == b"ORIGINAL_BYTES"
    assert not _tmp_files(root), f"leftover temp files after write failure: {_tmp_files(root)}"

    # Cache-hit path through run_dxc when os.replace fails.
    with mock.patch.dict(os.environ, {mod.ENV_VAR: str(cache_dir)}):
        mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(warm), str(src)], capture_output=True, check=True)
        with mock.patch.object(mod.os, "replace", side_effect=OSError("disk full")):
            try:
                mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(target), str(src)], capture_output=True, check=True)
            except OSError:
                pass
            else:
                raise AssertionError("cache-hit run_dxc did not propagate os.replace OSError")
    assert target.read_bytes() == b"ORIGINAL_BYTES", "cache-hit os.replace fail altered prior output"
    assert not _tmp_files(root), f"leftover temp files after cache-hit replace failure: {_tmp_files(root)}"


def _test_fo_dir_and_symlink_refusal(mod, root, compiler):
    out_dir = Path(root) / "dir_target"
    out_dir.mkdir()
    try:
        mod._write_file(str(out_dir), b"data")
    except OSError:
        pass
    else:
        raise AssertionError("_write_file must refuse directory target")
    assert not _tmp_files(root)

    victim = Path(root) / "victim.cso"
    victim.write_bytes(b"VICTIM_INTACT")
    link_out = Path(root) / "link_out.cso"
    real_symlink = _make_symlink(str(victim), str(link_out))
    if real_symlink:
        try:
            mod._write_file(str(link_out), b"OVERWRITE")
        except OSError:
            pass
        else:
            raise AssertionError("_write_file must refuse symlink target")
        assert victim.read_bytes() == b"VICTIM_INTACT", "symlink target file was modified"
        assert not _tmp_files(root)
    else:
        with mock.patch.object(mod.os.path, "islink", side_effect=lambda p: os.fspath(p) == str(link_out)):
            try:
                mod._write_file(str(link_out), b"OVERWRITE")
            except OSError:
                pass
            else:
                raise AssertionError("_write_file must refuse symlink target (mocked)")
            assert not _tmp_files(root)

    # Cache-hit run_dxc with -Fo pointing at symlink must refuse and preserve victim.
    cache_dir = Path(root) / "cache_link_fo"
    src = Path(root) / "link_fo.hlsl"
    src.write_bytes(b"void main(){}")
    warm = Path(root) / "warm_fo.cso"
    with mock.patch.dict(os.environ, {mod.ENV_VAR: str(cache_dir)}):
        mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(warm), str(src)], capture_output=True, check=True)
        if real_symlink:
            try:
                mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(link_out), str(src)],
                            capture_output=True, check=True)
            except OSError:
                pass
            else:
                raise AssertionError("cache-hit run_dxc must refuse symlink -Fo")
            assert victim.read_bytes() == b"VICTIM_INTACT"
            assert not _tmp_files(root)
        # -Fo pointing to directory bypasses cache via _plan and fails in compiler.
        before = _calls(root)
        res_dir = mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(out_dir), str(src)], capture_output=True)
        assert res_dir.returncode != 0
        assert _calls(root) == before + 1, "-Fo directory must bypass cache to compiler"
        assert len(list(cache_dir.glob("*.json"))) == 1


def _test_symlink_and_file_cache_dir_fallback(mod, root, compiler):
    src = Path(root) / "fallback.hlsl"
    src.write_bytes(b"void main(){}")

    # 1. File-backed cache_dir (and nested child under file) -> uncached fallback without crashing.
    file_cache = Path(root) / "file_cache"
    file_cache.write_bytes(b"NOT_A_DIR")
    for cache_val in (str(file_cache), str(file_cache / "nested")):
        before = _calls(root)
        with mock.patch.dict(os.environ, {mod.ENV_VAR: cache_val}):
            r1 = mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(Path(root) / "fb1.cso"), str(src)],
                             capture_output=True, check=True)
            r2 = mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(Path(root) / "fb2.cso"), str(src)],
                             capture_output=True, check=True)
        assert r1.returncode == 0 and r2.returncode == 0
        assert _calls(root) == before + 2, f"file-backed cache_dir {cache_val} must run uncached"
        assert file_cache.read_bytes() == b"NOT_A_DIR"

    # 2. Symlink-backed cache_dir -> uncached fallback, real target stays empty.
    real_cache = Path(root) / "real_cache_target"
    real_cache.mkdir()
    link_cache = Path(root) / "link_cache"
    if _make_symlink(str(real_cache), str(link_cache)):
        before = _calls(root)
        with mock.patch.dict(os.environ, {mod.ENV_VAR: str(link_cache)}):
            mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(Path(root) / "sl1.cso"), str(src)],
                        capture_output=True, check=True)
            mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(Path(root) / "sl2.cso"), str(src)],
                        capture_output=True, check=True)
        assert _calls(root) == before + 2, "symlink cache_dir must run uncached"
        assert list(real_cache.iterdir()) == [], "symlink cache_dir wrote to target"

        # Trailing slash on symlink cache_dir must still be rejected.
        before = _calls(root)
        with mock.patch.dict(os.environ, {mod.ENV_VAR: str(link_cache) + os.sep}):
            mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(Path(root) / "sl3.cso"), str(src)],
                        capture_output=True, check=True)
        assert _calls(root) == before + 1
        assert list(real_cache.iterdir()) == []
    else:
        before = _calls(root)
        with mock.patch.object(mod.os.path, "islink", side_effect=lambda p: os.fspath(p) == str(link_cache)):
            with mock.patch.dict(os.environ, {mod.ENV_VAR: str(link_cache)}):
                mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(Path(root) / "sl1.cso"), str(src)],
                            capture_output=True, check=True)
                mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(Path(root) / "sl2.cso"), str(src)],
                            capture_output=True, check=True)
        assert _calls(root) == before + 2


def _test_multi_output_fc_fd(mod, root, compiler):
    cache_dir = Path(root) / "cache_multi"
    src = Path(root) / "multi.hlsl"
    src.write_bytes(b"float4 main() : SV_Target { return 0; }")
    fo1, fc1, fd1 = Path(root) / "m1.cso", Path(root) / "m1.asm", Path(root) / "m1.pdb"
    fo2, fc2, fd2 = Path(root) / "m2.cso", Path(root) / "m2.asm", Path(root) / "m2.pdb"
    before = _calls(root)
    with mock.patch.dict(os.environ, {mod.ENV_VAR: str(cache_dir)}):
        r1 = mod.run_dxc(compiler, ["-T", "ps_6_0", "-Fo", str(fo1), "-Fc", str(fc1), "-Fd", str(fd1), str(src)],
                         capture_output=True, text=True, check=True)
        r2 = mod.run_dxc(compiler, ["-T", "ps_6_0", "-Fo", str(fo2), "-Fc", str(fc2), "-Fd", str(fd2), str(src)],
                         capture_output=True, text=True, check=True)
    assert _calls(root) == before + 1, "multi-output -Fc/-Fd missed cache"
    assert fo2.read_bytes().startswith(b"FO:")
    assert fc2.read_bytes().startswith(b"FC:")
    assert fd2.read_bytes().startswith(b"FD:")
    assert str(fc2) in r2.stdout and str(fd2) in r2.stdout and str(fo2) in r2.stdout
    assert str(fo1) not in r2.stdout and str(fc1) not in r2.stdout
    assert r1.returncode == r2.returncode == 0
    assert not _tmp_files(root)


def _test_corrupt_cache_miss(mod, root, compiler):
    cache_dir = Path(root) / "cache_corrupt"
    src = Path(root) / "corrupt.hlsl"
    src.write_bytes(b"void main(){ /*corrupt*/ }")
    out1 = Path(root) / "c1.cso"
    out2 = Path(root) / "c2.cso"
    with mock.patch.dict(os.environ, {mod.ENV_VAR: str(cache_dir)}):
        mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(out1), str(src)], capture_output=True, check=True)
        entries = list(cache_dir.glob("*.json"))
        assert len(entries) == 1
        entry = entries[0]
        # Invalid JSON -> treated as cache miss and overwritten with valid entry.
        entry.write_text("{not-json", encoding="utf-8")
        before = _calls(root)
        mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(out2), str(src)], capture_output=True, check=True)
        assert _calls(root) == before + 1
        assert out2.read_bytes().startswith(b"FO:")
        # Bad hex payload -> treated as cache miss.
        rec = json.loads(entry.read_text(encoding="utf-8"))
        rec["files"]["-Fo"] = "zzzz"
        entry.write_text(json.dumps(rec), encoding="utf-8")
        before = _calls(root)
        out3 = Path(root) / "c3.cso"
        mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(out3), str(src)], capture_output=True, check=True)
        assert _calls(root) == before + 1
        assert out3.read_bytes().startswith(b"FO:")


def _test_include_and_i_bypass(mod, root, compiler):
    cache_dir = Path(root) / "cache_bypass"
    inc_src = Path(root) / "inc.hlsl"
    inc_src.write_bytes(b'#include "defs.hlsli"\nvoid main(){}')
    plain_src = Path(root) / "plain.hlsl"
    plain_src.write_bytes(b"void main(){}")
    with mock.patch.dict(os.environ, {mod.ENV_VAR: str(cache_dir)}):
        before = _calls(root)
        mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(Path(root) / "i1.cso"), str(inc_src)], capture_output=True)
        mod.run_dxc(compiler, ["-T", "cs_6_0", "-Fo", str(Path(root) / "i2.cso"), str(inc_src)], capture_output=True)
        mod.run_dxc(compiler, ["-T", "cs_6_0", "-I", str(root), "-Fo", str(Path(root) / "d1.cso"), str(plain_src)],
                    capture_output=True)
        mod.run_dxc(compiler, ["-T", "cs_6_0", "-I", str(root), "-Fo", str(Path(root) / "d2.cso"), str(plain_src)],
                    capture_output=True)
        assert _calls(root) == before + 4, "#include or -I must bypass cache every time"
        assert not cache_dir.exists() or list(cache_dir.glob("*.json")) == []


def _test_proxy(mod, root, compiler):
    cache_dir = Path(root) / "cache_proxy"
    src = Path(root) / "proxy.hlsl"
    src.write_bytes(b"void main(){}")
    proxy = mod.with_dxc_cache(subprocess)
    with mock.patch.dict(os.environ, {mod.ENV_VAR: str(cache_dir)}):
        before = _calls(root)
        proxy.run([compiler, "-T", "cs_6_0", "-Fo", str(Path(root) / "p1.cso"), str(src)], capture_output=True)
        proxy.run([compiler, "-T", "cs_6_0", "-Fo", str(Path(root) / "p2.cso"), str(src)], capture_output=True)
        assert _calls(root) == before + 1
    sentinel = subprocess.CompletedProcess(["other"], 0)
    with mock.patch.object(subprocess, "run", return_value=sentinel) as patched:
        got = proxy.run(["other", "--version"])
        assert got is sentinel and patched.call_count == 1


def run_suite(mod):
    tests = [
        _test_atomic_restore_and_zero_temps,
        _test_replace_fail_preserves_output_and_cleans_temps,
        _test_fo_dir_and_symlink_refusal,
        _test_symlink_and_file_cache_dir_fallback,
        _test_multi_output_fc_fd,
        _test_corrupt_cache_miss,
        _test_include_and_i_bypass,
        _test_proxy,
    ]
    for fn in tests:
        with tempfile.TemporaryDirectory() as root:
            compiler = _make_rig(root)
            fn(mod, root, compiler)
    return len(tests)


def _run_mutations():
    source = DXC_CACHE_PATH.read_text(encoding="utf-8")
    mutants = [
        (
            "drop_symlink_target_guard",
            "    if os.path.islink(target) or os.path.islink(stripped):\n"
            "        raise OSError(errno.ELOOP, \"refusing symlink target\", target)\n",
            "    if False:\n        pass\n",
            _test_fo_dir_and_symlink_refusal,
        ),
        (
            "non_atomic_write_file",
            "        os.replace(temp, target)",
            "        with open(target, 'wb') as direct: direct.write(data)\n        os.unlink(temp)",
            _test_replace_fail_preserves_output_and_cleans_temps,
        ),
        (
            "drop_tmp_cleanup",
            "        if not closed:\n            os.close(handle)\n"
            "        if os.path.exists(temp):\n            os.unlink(temp)",
            "        if not closed:\n            os.close(handle)",
            _test_replace_fail_preserves_output_and_cleans_temps,
        ),
        (
            "drop_cache_dir_symlink_and_file_guard",
            "    if _valid_cache_dir(cache_dir) and not extra:",
            "    if cache_dir and not extra:",
            _test_symlink_and_file_cache_dir_fallback,
        ),
        (
            "drop_include_bypass",
            "    if INCLUDE_RE.search(source_bytes):",
            "    if False:",
            _test_include_and_i_bypass,
        ),
    ]
    with tempfile.TemporaryDirectory() as mut_dir:
        for idx, (name, needle, repl, probe) in enumerate(mutants):
            assert source.count(needle) == 1, f"mutant needle not unique: {name}"
            mut_path = Path(mut_dir) / f"dxc_cache_mut_{idx}.py"
            mut_path.write_text(source.replace(needle, repl, 1), encoding="utf-8")
            mod = _load_module(mut_path, f"dxc_cache_mut_{idx}")
            with tempfile.TemporaryDirectory() as root:
                compiler = _make_rig(root)
                try:
                    probe(mod, root, compiler)
                except (AssertionError, OSError, SystemExit):
                    continue
                raise SystemExit(f"dxc_cache_unit_check: mutant {name} survived!")
    return len(mutants)


def main():
    mod = _load_module(DXC_CACHE_PATH, "dxc_cache")
    passed = run_suite(mod)
    killed = _run_mutations()
    print(f"dxc_cache_unit_check: {passed}/{passed} unit checks passed, {killed}/{killed} mutants killed")


if __name__ == "__main__":
    main()
