"""Run a compiler-built RDS5 directory reader, its mapped PE, and its AOT image on the same provider."""
from collections import defaultdict
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from startup_storage_directory_batch import DISTANT, PAGE, SHIFT, ranges, write_compressed_fixture
from startup_xg_bc_chain_check import block
from texture_descriptors import iter_boundaries, read_tables


def solid_pages():
    groups = defaultdict(list)
    for red in range(2, 252, 2):
        color = bytes((red, 16, 48, 254))
        page = block(98, color) * 256
        stream = zlib.compress(page, 9)
        groups[len(stream)].append((color, page, stream))
    # Only changing the content distinguishes the shifted first record.
    group = max(groups.values(), key=len)
    different_length = next(items[0] for items in groups.values() if len(items[0][2]) != len(group[0][2]))
    return group[0], different_length, group[-1]


def runners(mapper, native, lifted, provider):
    mapped = [sys.executable, "-B", Path(__file__).with_name("archive_check.py"), "--mapped", mapper, native, provider]
    return {"native": [native], "pe_map": mapped, "AOT": [lifted]}


def check(mapper, native, lifted, provider):
    first, second, decoy = solid_pages()
    commands = runners(mapper, native, lifted, provider)
    adapter = None
    env = dict(os.environ, X2W_XMEM_PERMISSIVE="1")
    with tempfile.TemporaryDirectory() as workspace:
        for records in (1, 16384):
            case = Path(workspace) / str(records)
            case.mkdir()
            path, alternate = (case / name for name in ("fixture.ttpf", "alternate.ttpf"))
            write_compressed_fixture(path, records, [p[2] for p in (first, second, decoy)])
            write_compressed_fixture(alternate, records, [p[2] for p in (decoy, second, first)])
            directory = ranges(path)
            assert not path.samefile(alternate) and ranges(alternate) == directory
            with path.open("rb") as source:
                metadata, compressed, *_ = iter_boundaries(read_tables(source))
            first_page = compressed.logical // PAGE
            base, length = directory[0]
            assert metadata.logical == 0 and directory[DISTANT] == (base + SHIFT, length)
            assert length != directory[1][1]
            # Each tuple specifies the logical page indices, an injected first-offset delta,
            # the input file selector, the expected two decoded pages, and the process result.
            cases = (
                ("correct", 0, 1, 0, 0, first, second, 0),
                ("later records", 1, DISTANT, 0, 0, second, decoy, 0),
                ("wrong file", 0, 1, 0, 1, decoy, second, 0),
                ("wrong record", DISTANT, 1, 0, 0, decoy, second, 0),
                ("wrong offset", 0, 1, SHIFT, 0, decoy, second, 0),
                ("EOF", 0, 1, path.stat().st_size - base, 0, None, second, 6),
                ("invalid count", 0, 1, 0, 0, None, None, 3),
                ("terminal beyond EOF", 0, 1, 0, 0, None, None, 3),
                ("missing second", 0, len(directory), 0, 0, None, None, 3),
            )
            corrupt = {}
            for fault in ("invalid count", "terminal beyond EOF"):
                bad_directory = case / fault
                bad_directory.mkdir()
                corrupt[fault] = bad_directory
                shutil.copyfile(path, bad_directory / path.name)
                with (bad_directory / path.name).open("r+b") as invalid:
                    invalid.seek(4)
                    entry_count, = struct.unpack("<I", invalid.read(4))
                    if fault == "invalid count":
                        invalid.seek(4)
                        invalid.write(struct.pack("<I", 0xffffffff))
                    else:
                        position = 16 + (entry_count - 1) * 8
                        invalid.seek(position)
                        terminal, = struct.unpack("<Q", invalid.read(8))
                        invalid.seek(position)
                        invalid.write(struct.pack("<Q", terminal + 1))  # Last physical boundary moves by 256 bytes.
            for mode, i, j, bias, selector, a, b, exit_code in cases:
                for runner, command in commands.items():
                    root = corrupt.get(mode, case)
                    folder = root / (mode + "-" + runner)
                    folder.mkdir()
                    selected_file = root / (alternate.name if selector else path.name)
                    inputs = struct.pack("<IIQI", first_page + i, first_page + j, bias, selector)
                    child = subprocess.run(command, input=inputs, cwd=folder, env=env, capture_output=True, timeout=12)
                    if child.returncode == 77 and runner == "native" and env.get("X2W_SHIM_DRAW_ADAPTER") == "hardware":
                        return 77
                    assert child.returncode == exit_code, (runner, mode, child.returncode, child.stderr)
                    report = folder / "ranges.txt"
                    pages = folder / "pages.bin"
                    rendered = folder / "pixels.bin"
                    if exit_code == 3:
                        assert not any(p.exists() for p in (report, pages, rendered)), (runner, mode, "unexpected queue")
                        continue
                    trace = report.read_text().splitlines()
                    pos, size = directory[i]
                    next_pos, next_size = directory[j]
                    assert trace.pop() == "storage release 0 0 0"
                    expected = [f"file {selected_file.stat().st_ino & 0xffffffffffffffff:016x}",
                                # only the EOF case has no destination page: ERROR_HANDLE_EOF
                                f"request {i+1} {pos+bias} {size} {PAGE} 0", f"request {j+1} {next_pos} {next_size} {PAGE} 0",
                                "status 00000000" if a else "status 80070026", "status 00000000"]
                    assert trace[:5] == expected, (runner, mode, trace, expected)
                    assert pages.read_bytes() == ((a[1] if a else b"\xa5" * PAGE) + b[1]), (runner, mode, "BC pages")
                    if a:
                        assert trace[5].startswith("adapter ")
                        if adapter is None:
                            adapter = trace[5]
                        assert trace[5:] == [adapter, "fence 1", "release 0"], (runner, mode, trace[5:])
                        assert rendered.read_bytes() == a[0] * 8192 + b[0] * 8192, (runner, mode, "RGBA")
                    else:
                        assert len(trace) == 5 and not rendered.exists()
                print(f"{records=} {mode}: native, mapped, AOT matched exit {exit_code}")
    print(adapter)
    return 0


if __name__ == "__main__":
    sys.exit(check(*[Path(arg).resolve() for arg in sys.argv[1:]]))
