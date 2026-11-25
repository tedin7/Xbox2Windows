"""Identify the first ID 16 writer and reader in the pinned default dispatch graph."""
import argparse
import os
from pathlib import Path
import struct
import tempfile

import pefile

from ffx_version_query import UPSCALER, checked_image, lower, region
from resource_inventory import parts


def dispatch_slices(pe):
    body = lower(region(pe, 0x13278, 0x132cc),
                 ["uint8_t desc[0x189]{}; desc[0x188] = sharpen;",
                  "r14 = reinterpret_cast<word>(desc); r12 = reinterpret_cast<word>(&provider);"],
                 {}, {}, {0x132cc: "return;"})
    code = "\nvoid dispatch_aliases(RecurrentProvider& provider, bool sharpen) {\n" + body + "\n}\n"
    # Output size lives in the graph builder's stack frame. Its list header at
    # r13 is read during this block, before the new three dimensions are appended.
    writer = lower(region(pe, 0x7e6a, 0x7eaf),
                   ["uint8_t local[512]{}; rbp = reinterpret_cast<word>(local);",
                    "store<uint32_t>(rbp+32,width); store<uint32_t>(rbp+36,height); r13=rsp;"],
                   {}, {}, {0x7eaf: "return load<DispatchGrid>(rbp+0x1f0);"})
    reader = lower(region(pe, 0x13869, 0x13880) + region(pe, 0x1388d, 0x13895),
                   ["uint8_t local[272]{}; rbp = reinterpret_cast<word>(local);",
                    "rsi=width; store<uint32_t>(rbp+0x104,height);"],
                   {}, {}, {0x13880: "goto block_1388d;",
                            0x13895: "return {uint32_t(r8), uint32_t(r9), uint32_t(frame[4])};"})
    return (code + "DispatchGrid producer_grid(uint32_t width, uint32_t height) {\n" + writer +
            "\n}\nDispatchGrid rcas_grid(uint32_t width, uint32_t height) {\n" + reader + "\n}\n")


def inspect(pe):
    base = pe.OPTIONAL_HEADER.ImageBase

    def pointer(rva):
        return pe.get_qword_at_rva(rva) - base

    def bindings(record, offset):
        table = record + offset
        count = pe.get_dword_at_rva(table)
        return {pe.get_string_at_rva(pointer(pointer(table + 8) + 8*i)).decode():
                pe.get_dword_at_rva(pointer(table + 16) + 4*i) for i in range(count)}

    def resource_id(name, start, count):
        for i in range(count):
            row = start + i * 0x84
            entry = pe.get_data(row + 4, 128)
            if len(entry) < 128:
                raise ValueError(f"Truncated resource descriptor at {row:#x}")
            if entry.decode("utf-16le").split("\0", 1)[0] == name:
                return pe.get_dword_at_rva(row)
        raise ValueError(f"Unknown binding {name}")

    writers = []
    predecessors = []
    for index in range(27):
        case = pe.get_dword_at_rva(0x260c0 + 4*index)
        load, = region(pe, case, case + 7)
        record = load.ip_rel_memory_address - 8
        if index in (24, 25):
            if bindings(record, 0xb8) != {"ScratchBuffer": 11}:
                raise ValueError(f"Unexpected pass {index} scratch binding")
            if index == 24 and bindings(record, 0x90) != {"InitializerBuffer": 18}:
                raise ValueError("Unexpected pass 24 model binding")
            predecessors.append((f"pass{index}", record, 64 if index == 24 else 32))
        srvs = bindings(record, 0x40)
        uavs = bindings(record, 0x68)
        if any(resource_id(name, 0x6fb60, 12) == 16 for name in srvs):
            raise ValueError(f"Unexpected earlier ID 16 reader in pass {index}")
        for name, slot in uavs.items():
            if resource_id(name, 0x72e00, 12) in (10, 16):
                writers.append((index, name, slot, record))
    if writers != [(26, "rw_mlsr_output_color", 2, 0xbd830)]:
        raise ValueError(f"Unexpected ID 16 writers {writers}")
    selector = pe.get_data(0x6fb20, 32)
    if len(selector) != 32:
        raise ValueError("Truncated RCAS permutation selector")
    if set(struct.unpack("<8I", selector)) != {0}:
        raise ValueError("Unexpected RCAS permutation selector")
    reader = 0xbc0d0
    if (bindings(reader, 0x40) != {"r_rcas_input": 18} or
            resource_id("r_rcas_input", 0x6fb60, 12) != 16 or
            bindings(reader, 0x68) != {"rw_rcas_output": 11} or bindings(reader, 0xe0)):
        raise ValueError("Unexpected RCAS shader bindings")

    for name, record, threads in [*predecessors, ("producer", writers[0][3], 256), ("rcas", reader, 64)]:
        blob = pointer(record + 8)
        container_size = pe.get_dword_at_rva(record)
        container = pe.get_data(blob, container_size)
        if len(container) != container_size:
            raise ValueError(f"Truncated {name} shader container")
        chunks = parts(container)
        psv0 = chunks.get(b"PSV0")
        if psv0 is None or len(psv0) < 52:
            raise ValueError(f"Missing or truncated {name} PSV0 section")
        if struct.unpack_from("<3I", psv0, 40) != (threads, 1, 1):
            raise ValueError(f"Unexpected {name} group size")
        native = chunks.get(b"ACCC")
        if native is None or len(native) < 16 or native[:4] != b"FSPX":
            raise ValueError(f"Missing or truncated {name} native compute shader")
        count, = struct.unpack_from("<I", native, 8)
        table_end = 16 + count * 16
        if table_end > len(native):
            raise ValueError(f"Truncated {name} FSPX section table")
        sections = struct.iter_unpack("<4I", native[16:table_end])
        matches = [(off, n) for tag, off, n, _ in sections if tag == 6]
        if len(matches) != 1:
            raise ValueError(f"Expected single {name} FSPX code section, found {len(matches)}")
        offset, size = matches[0]
        if offset < table_end or size <= 0x24 or offset + size > len(native):
            raise ValueError(f"Out-of-bounds {name} FSPX code section")
        code = native[offset + 0x24:offset + size]
        ends = [i + 4 for i in range(0, len(code) - 3, 4) if code[i:i + 4] == b"\0\0\x81\xbf"]
        if len(ends) != 1:
            raise ValueError(f"Expected single {name} s_endpgm terminator, found {len(ends)}")
        end = ends[0]
        print(f"{name} container={blob:#x} threads={threads} code_bytes={end}")
        yield name, code[:end]

    # The model initializer selector returns the RVA and byte count as a pair.
    init_size, init_base = region(pe, 0x78e5, 0x78f1)
    if init_size.immediate(1) != 0x20000 or init_base.ip_rel_memory_address != 0x4eee0:
        raise ValueError("Unexpected i8 initializer")
    init_data = pe.get_data(init_base.ip_rel_memory_address, init_size.immediate(1))
    if len(init_data) != 0x20000:
        raise ValueError("Truncated i8 initializer")
    yield "initializer", init_data


def _same_path(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and left.samefile(right)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if _same_path(args.output_directory, args.image):
            raise ValueError(f"Output directory collides with image: {args.output_directory}")
        if args.output_directory.exists() and not args.output_directory.is_dir():
            raise ValueError(f"Output directory is not a directory: {args.output_directory}")
        with checked_image(args.image, UPSCALER) as pe:
            programs = list(inspect(pe))
        outputs = [(args.output_directory / f"{name}.bin", code) for name, code in programs]
        for destination, _ in outputs:
            if _same_path(destination, args.image):
                raise ValueError(f"Output file collides with image: {destination}")
        args.output_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output_directory) as staging_dir:
            staging_root = Path(staging_dir)
            staged = []
            for destination, code in outputs:
                staging_path = staging_root / destination.name
                staging_path.write_bytes(code)
                staged.append((staging_path, destination))
            for staging_path, destination in staged:
                os.replace(staging_path, destination)
    except (OSError, ValueError, struct.error, pefile.PEFormatError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
