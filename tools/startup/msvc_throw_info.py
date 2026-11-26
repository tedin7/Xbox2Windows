#!/usr/bin/env python3
"""Decode the public MSVC x64 C++ exception record into a sanitized summary (#1758).

A startup run can end on a second-chance exception 0xE06D7363. Its ExceptionInformation holds a magic
number, the address of the thrown object, the address of the throw info and (x64) the image base the
throw info's relative addresses count from. The layout is the public one of the MSVC runtime
(ehdata.h):

  ExceptionInformation[0]   magic, 0x19930520 (0x19930521 and 0x19930522 are later versions)
  ExceptionInformation[1]   the thrown object
  ExceptionInformation[2]   _ThrowInfo
  ExceptionInformation[3]   image base (x64, NumberParameters >= 4)

  _ThrowInfo           int32 attributes @0, RVA unwind @4, RVA forward-compat @8, RVA _CatchableTypeArray @12
  _CatchableTypeArray  int32 count @0, then count RVAs of _CatchableType @4
  _CatchableType       int32 properties @0, RVA type_info @4, PMD displacement @8 (3 x int32),
                       int32 size @20, RVA copy constructor @24
  type_info            vtable @0, cache pointer @8, decorated name (NUL terminated) @16

The decoder takes the exception code, the parameters and a `read_mem(address, size) -> bytes | None` callback,
so it has no Win32 dependency and runs the same on Linux and Windows. It never returns an address, a path
or a name from the title: the record has the decorated name of a built-in scalar or of a public standard
library or COM type, `other` for any other class, and for a thrown 32-bit scalar the HRESULT as a public
name (E_INVALIDARG, DXGI_ERROR_DEVICE_REMOVED, ...) and its facility category. A failure-looking value that
is not a recognized public HRESULT is only classified as `unrecognized_failure`, never printed.
Uses only the Python 3 standard library.
"""

from __future__ import annotations

import re
import struct
from typing import Callable, Optional, Sequence

EXCEPTION_CODE_MSVC_CPP = 0xE06D7363
MAGICS = (0x19930520, 0x19930521, 0x19930522)
MAX_TYPES = 16  # a throw of a class lists its bases too; more than this is not a real throw info
MAX_NAME_BYTES = 96
NAME_CHUNK = 16
U64 = (1 << 64) - 1

ReadMem = Callable[[int, int], Optional[bytes]]

# Decorated names of the built-in 32-bit scalars: (description, signed).
SCALARS_32 = {".J": ("long", True), ".H": ("int", True), ".I": ("unsigned int", False), ".K": ("unsigned long", False)}
# Other built-in scalars, named but with no value read.
SCALARS_OTHER = {
    ".D": "char", ".C": "signed char", ".E": "unsigned char", ".F": "short", ".G": "unsigned short", ".M": "float",
    ".N": "double", ".O": "long double", ".X": "void", "._J": "__int64", "._K": "unsigned __int64", "._N": "bool",
}
# Public library classes whose decorated name is safe to report. Any other class is reported as "other".
PUBLIC_CLASSES = {
    ".?AV_com_error@@": "_com_error",
    ".?AVexception@std@@": "std::exception",
    ".?AVbad_alloc@std@@": "std::bad_alloc",
    ".?AVbad_array_new_length@std@@": "std::bad_array_new_length",
    ".?AVbad_cast@std@@": "std::bad_cast",
    ".?AVbad_function_call@std@@": "std::bad_function_call",
    ".?AVlogic_error@std@@": "std::logic_error",
    ".?AVinvalid_argument@std@@": "std::invalid_argument",
    ".?AVlength_error@std@@": "std::length_error",
    ".?AVout_of_range@std@@": "std::out_of_range",
    ".?AVruntime_error@std@@": "std::runtime_error",
    ".?AVsystem_error@std@@": "std::system_error",
    ".?AVfailure@ios_base@std@@": "std::ios_base::failure",
}
NAME_FORM = re.compile(r"^\.[A-Za-z0-9_?@$]{1,%d}$" % (MAX_NAME_BYTES - 1))

HRESULT_NAMES = {
    0x00000000: "S_OK", 0x00000001: "S_FALSE",
    0x80004001: "E_NOTIMPL", 0x80004002: "E_NOINTERFACE", 0x80004003: "E_POINTER", 0x80004004: "E_ABORT",
    0x80004005: "E_FAIL", 0x8000FFFF: "E_UNEXPECTED", 0x8000000A: "E_PENDING",
    0x80070005: "E_ACCESSDENIED", 0x80070006: "E_HANDLE", 0x8007000E: "E_OUTOFMEMORY", 0x80070057: "E_INVALIDARG",
    0x80070002: "HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND)", 0x80070003: "HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND)",
    0x80070008: "HRESULT_FROM_WIN32(ERROR_NOT_ENOUGH_MEMORY)", 0x8007000D: "HRESULT_FROM_WIN32(ERROR_INVALID_DATA)",
    0x80070216: "HRESULT_FROM_WIN32(ERROR_ARITHMETIC_OVERFLOW)", 0x800705AF: "HRESULT_FROM_WIN32(ERROR_COMMITMENT_LIMIT)",
    0x887A0001: "DXGI_ERROR_INVALID_CALL", 0x887A0002: "DXGI_ERROR_NOT_FOUND", 0x887A0003: "DXGI_ERROR_MORE_DATA",
    0x887A0004: "DXGI_ERROR_UNSUPPORTED", 0x887A0005: "DXGI_ERROR_DEVICE_REMOVED", 0x887A0006: "DXGI_ERROR_DEVICE_HUNG",
    0x887A0007: "DXGI_ERROR_DEVICE_RESET", 0x887A000A: "DXGI_ERROR_WAS_STILL_DRAWING",
    0x887A0020: "DXGI_ERROR_DRIVER_INTERNAL_ERROR", 0x887A0026: "DXGI_ERROR_ACCESS_LOST", 0x887A0027: "DXGI_ERROR_WAIT_TIMEOUT",
    0x887E0001: "D3D12_ERROR_ADAPTER_NOT_FOUND", 0x887E0002: "D3D12_ERROR_DRIVER_VERSION_MISMATCH",
    0x887E0003: "D3D12_ERROR_INVALID_REDIST",
}
FACILITY_CATEGORIES = {0x007: "win32", 0x87A: "dxgi", 0x87E: "d3d12"}
UNRECOGNIZED_HRESULT = "unrecognized_failure"
SMALL_VALUE_LIMIT = 0xFFFF # a non-HRESULT scalar above this could be anything, so no number is reported


def hresult_name(value: int) -> str:
    """The public name of an HRESULT, or HRESULT(0x...) for one that has none."""
    value &= 0xFFFFFFFF
    return HRESULT_NAMES.get(value, "HRESULT(0x%08X)" % value)


def hresult_category(value: int) -> str:
    """The facility category of a failure HRESULT: win32, dxgi, d3d12 or other; success for a non-failure code."""
    value &= 0xFFFFFFFF
    if not value & 0x80000000:
        return "success"
    return FACILITY_CATEGORIES.get((value >> 16) & 0x1FFF, "other")


def _read(read_mem: ReadMem, address: int, size: int) -> Optional[bytes]:
    """read_mem with the address checked first; None for a failed, short or out-of-range read."""
    if address < 0 or size <= 0 or address + size > U64 + 1:
        return None
    try:
        data = read_mem(address, size)
    except Exception:  # a callback that raises is a read that failed
        return None
    return bytes(data) if data is not None and len(data) == size else None


def _read_slowly(read_mem: ReadMem, address: int) -> Optional[bytes]:
    """Up to NAME_CHUNK bytes one at a time, ending at the first NUL; None when not even one byte is readable."""
    out = b""
    for offset in range(NAME_CHUNK):
        byte = _read(read_mem, address + offset, 1)
        if byte is None:
            break
        out += byte
        if byte == b"\0":
            return out
    return out if out else None


def _read_name(read_mem: ReadMem, address: int) -> Optional[str]:
    """The NUL terminated decorated name at address, read in short pieces; None when unreadable or not a name."""
    raw = b""
    while len(raw) < MAX_NAME_BYTES:
        piece = _read(read_mem, address + len(raw), NAME_CHUNK)
        if piece is None:
            piece = _read_slowly(read_mem, address + len(raw))  # the name may end just before unreadable memory
        if piece is None:
            return None
        end = piece.find(b"\0")
        if end >= 0:
            raw += piece[:end]
            break
        raw += piece
    else:
        return None
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        return None
    return text if NAME_FORM.match(text) else None


def _classify(name: str) -> dict:
    """A reportable form of a decorated type name: built-in and public names as they are, anything else as 'other'."""
    if name in SCALARS_32:
        return {"type": name, "kind": "scalar", "c_type": SCALARS_32[name][0]}
    if name in SCALARS_OTHER:
        return {"type": name, "kind": "scalar", "c_type": SCALARS_OTHER[name]}
    if name in PUBLIC_CLASSES:
        return {"type": name, "kind": "class", "c_type": PUBLIC_CLASSES[name]}
    return {"type": "other", "kind": "class" if name.startswith(".?A") else "other"}


def _refused(status: str) -> dict:
    return {"status": status}


def decode_throw(exception_code: int, params: Sequence[int], read_mem: ReadMem, max_types: int = MAX_TYPES) -> dict:
    """Decode one exception record. The result always has a 'status'; with status 'ok' it also has
    'types' (one entry per catchable type), 'type_count' and, for a thrown 32-bit scalar, 'value'.

    status: ok, not_msvc_exception, missing_parameters, bad_magic, no_throw_info, unreadable, bad_throw_info,
    too_many_types. The record holds no address and no path."""
    if exception_code & 0xFFFFFFFF != EXCEPTION_CODE_MSVC_CPP:
        return _refused("not_msvc_exception")
    if len(params) < 4:
        return _refused("missing_parameters")  # x64 passes the image base as the fourth parameter
    magic, thrown, throw_info, image_base = (int(p) & U64 for p in params[:4])
    if magic not in MAGICS:
        return _refused("bad_magic")
    if not throw_info or not image_base:
        return _refused("no_throw_info")

    head = _read(read_mem, throw_info, 16)
    if head is None:
        return _refused("unreadable")
    attributes, _unwind, _forward, array_rva = struct.unpack("<iIII", head)
    if not array_rva:
        return _refused("bad_throw_info")
    array = image_base + array_rva
    count_bytes = _read(read_mem, array, 4)
    if count_bytes is None:
        return _refused("unreadable")
    (count,) = struct.unpack("<i", count_bytes)
    if count < 0:
        return _refused("bad_throw_info")
    if count > max_types:
        return _refused("too_many_types")
    rvas = _read(read_mem, array + 4, 4 * count) if count else b""
    if rvas is None:
        return _refused("unreadable")

    types = []
    sizes = []
    for (rva,) in struct.iter_unpack("<I", rvas):
        if not rva:
            return _refused("bad_throw_info")
        entry = _read(read_mem, image_base + rva, 28)
        if entry is None:
            return _refused("unreadable")
        type_rva, size = struct.unpack_from("<I", entry, 4)[0], struct.unpack_from("<i", entry, 20)[0]
        if not type_rva:
            return _refused("bad_throw_info")
        name = _read_name(read_mem, image_base + type_rva + 16)
        if name is None:
            return _refused("unreadable")
        types.append(_classify(name))
        sizes.append(size)

    record = {"status": "ok", "type_count": len(types), "types": types, "attributes": attributes & 0xFF}
    if types and types[0]["type"] in SCALARS_32 and thrown and sizes[0] == 4:
        c_type, signed = SCALARS_32[types[0]["type"]]
        raw = _read(read_mem, thrown, 4)
        if raw is not None:
            value = struct.unpack("<i" if signed else "<I", raw)[0]
            record["value"] = _describe_value(c_type, value & 0xFFFFFFFF)
    return record


def _describe_value(c_type: str, bits: int) -> dict:
    """What a thrown 32-bit scalar says: a failure HRESULT by name and facility, a small number, or just 'large'."""
    if bits in HRESULT_NAMES:
        return {"c_type": c_type, "hresult": HRESULT_NAMES[bits], "category": hresult_category(bits),
                "code": "0x%08X" % bits}
    if bits & 0x80000000:
        # Not a recognized public HRESULT: it can be private application data, so no bits, code or facility.
        return {"c_type": c_type, "hresult": UNRECOGNIZED_HRESULT, "category": UNRECOGNIZED_HRESULT}
    if bits <= SMALL_VALUE_LIMIT:
        return {"c_type": c_type, "number": bits, "category": "success" if bits == 0 else "number"}
    return {"c_type": c_type, "category": "number", "number": None}


def describe(record: dict) -> str:
    """One log line for a decoded record, made only of the allowlisted fields of the record."""
    if record.get("status") != "ok":
        return "msvc-throw %s" % record.get("status", "unknown")
    kinds = ",".join(t["type"] for t in record["types"][:4]) or "none"
    line = "msvc-throw ok types=%s count=%d" % (kinds, record["type_count"])
    value = record.get("value")
    if value:
        if "hresult" in value:
            line += " %s=%s category=%s" % (value["c_type"].replace(" ", "_"), value["hresult"], value["category"])
        elif value.get("number") is not None:
            line += " %s=%d" % (value["c_type"].replace(" ", "_"), value["number"])
        else:
            line += " %s=large" % value["c_type"].replace(" ", "_")
    return line
