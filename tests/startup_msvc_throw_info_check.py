"""Tests for tools/startup/msvc_throw_info.py: synthetic exception records, no title data, no Win32."""

from __future__ import annotations

import json
from pathlib import Path
import re
import struct
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "startup"))
import msvc_throw_info as m  # noqa: E402

CODE = 0xE06D7363
MAGIC = 0x19930520
BASE = 0x180000000  # synthetic image base
THROW_INFO = BASE + 0x2000
ARRAY_RVA = 0x2100
THROWN = 0x2_0000_0000

failures: list[str] = []


def check(label: str, condition: bool, detail: object = "") -> None:
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail != "":
            print(f"  {detail}")


class Memory:
    """A sparse synthetic address space; a read touching an unmapped byte fails."""

    def __init__(self) -> None:
        self.bytes: dict[int, int] = {}

    def put(self, address: int, data: bytes) -> None:
        for offset, value in enumerate(data):
            self.bytes[address + offset] = value

    def read(self, address: int, size: int):
        if any(address + i not in self.bytes for i in range(size)):
            return None
        return bytes(self.bytes[address + i] for i in range(size))


def build(types, value: bytes | None = struct.pack("<i", -2147024809), sizes=None, count=None, array_rva=ARRAY_RVA,
          name_terminated=True):
    """A throw of an object whose catchable types have these decorated names. Returns (params, memory)."""
    mem = Memory()
    count = len(types) if count is None else count
    mem.put(THROW_INFO, struct.pack("<iIII", 0, 0, 0, array_rva))
    array = BASE + array_rva
    rvas = [0x2200 + 0x40 * i for i in range(len(types))]
    mem.put(array, struct.pack("<i", count) + b"".join(struct.pack("<I", r) for r in rvas))
    for index, name in enumerate(types):
        size = 4 if sizes is None else sizes[index]
        type_rva = 0x3000 + 0x80 * index
        mem.put(BASE + rvas[index], struct.pack("<iIiiiiI", 0, type_rva, 0, -1, 0, size, 0))
        mem.put(BASE + type_rva, bytes(16) + name.encode("ascii") + (b"\0" if name_terminated else b""))
    if value is not None:
        mem.put(THROWN, value)
    return [MAGIC, THROWN, THROW_INFO, BASE], mem


def decode(params, mem):
    return m.decode_throw(CODE, params, mem.read)


def long_value(hresult: int):
    return struct.pack("<I", hresult & 0xFFFFFFFF)


# 1. A thrown long: the HRESULT is named and sorted into its facility.
for hresult, name, category in (
    (0x80070057, "E_INVALIDARG", "win32"),
    (0x8007000E, "E_OUTOFMEMORY", "win32"),
    (0x80004001, "E_NOTIMPL", "other"),
    (0x80004005, "E_FAIL", "other"),
    (0x887A0005, "DXGI_ERROR_DEVICE_REMOVED", "dxgi"),
    (0x887A0006, "DXGI_ERROR_DEVICE_HUNG", "dxgi"),
    (0x887E0002, "D3D12_ERROR_DRIVER_VERSION_MISMATCH", "d3d12"),
    (0x800705AF, "HRESULT_FROM_WIN32(ERROR_COMMITMENT_LIMIT)", "win32"),
    (0x80041234, "HRESULT(0x80041234)", "other"),
    (0x887A9999, "HRESULT(0x887A9999)", "dxgi"),
):
    params, mem = build([".J"], long_value(hresult))
    record = decode(params, mem)
    value = record.get("value", {})
    check(f"long {hresult:#010x} is {name}", record["status"] == "ok" and value.get("hresult") == name
          and value.get("category") == category and value.get("code") == "0x%08X" % hresult, record)
    check(f"long {hresult:#010x}: the type is the built-in long", record["types"][0]["type"] == ".J"
          and record["types"][0]["c_type"] == "long")

# 2. Other 32-bit scalars, and values that are not an HRESULT.
params, mem = build([".H"], struct.pack("<i", -2147467259))
check("a thrown int is read signed", decode(params, mem)["value"]["hresult"] == "E_FAIL")
params, mem = build([".I"], struct.pack("<I", 0x887A0005))
check("a thrown unsigned int is read unsigned", decode(params, mem)["value"]["hresult"] == "DXGI_ERROR_DEVICE_REMOVED")
params, mem = build([".J"], struct.pack("<i", 5))
check("a small number is reported as a number", decode(params, mem)["value"] == {"c_type": "long", "number": 5, "category": "number"})
params, mem = build([".J"], struct.pack("<i", 0x12345678))
check("a large positive number is not reported", decode(params, mem)["value"]["number"] is None
      and "12345678" not in json.dumps(decode(params, mem)))
params, mem = build([".J"], long_value(0))
check("zero is S_OK", decode(params, mem)["value"]["hresult"] == "S_OK")

# 3. Records that are not decoded.
params, mem = build([".J"])
check("another exception code", m.decode_throw(0xC0000005, params, mem.read)["status"] == "not_msvc_exception")
check("the code is compared on 32 bits", m.decode_throw(0xFFFFFFFFE06D7363, params, mem.read)["status"] == "ok")
check("three parameters (no image base)", m.decode_throw(CODE, params[:3], mem.read)["status"] == "missing_parameters")
check("no parameters", m.decode_throw(CODE, [], mem.read)["status"] == "missing_parameters")
check("wrong magic", m.decode_throw(CODE, [0x12345678] + params[1:], mem.read)["status"] == "bad_magic")
for magic in (0x19930521, 0x19930522):
    check(f"magic {magic:#x} is accepted", m.decode_throw(CODE, [magic] + params[1:], mem.read)["status"] == "ok")
check("null throw info (a rethrow)", m.decode_throw(CODE, [MAGIC, THROWN, 0, BASE], mem.read)["status"] == "no_throw_info")
check("null image base", m.decode_throw(CODE, [MAGIC, THROWN, THROW_INFO, 0], mem.read)["status"] == "no_throw_info")

# 4. Reads that fail, at every step.
params, mem = build([".J"])
for label, address in (("throw info", THROW_INFO), ("catchable array", BASE + ARRAY_RVA), ("catchable type", BASE + 0x2200),
                       ("type_info name", BASE + 0x3000 + 16)):
    broken = Memory()
    broken.bytes = {a: v for a, v in mem.bytes.items() if not address <= a < address + 4}
    check(f"unreadable {label}", decode(params, broken)["status"] == "unreadable")
broken = Memory()
broken.bytes = {a: v for a, v in mem.bytes.items() if not THROWN <= a < THROWN + 4}
record = decode(params, broken)
check("an unreadable thrown object keeps the types and drops the value", record["status"] == "ok" and "value" not in record)
check("a callback that returns nothing", m.decode_throw(CODE, params, lambda a, n: None)["status"] == "unreadable")
check("a callback that returns too few bytes", m.decode_throw(CODE, params, lambda a, n: b"\0" * (n - 1))["status"] == "unreadable")


def raising(_address, _size):
    raise OSError("read failed")


check("a callback that raises", m.decode_throw(CODE, params, raising)["status"] == "unreadable")

# 5. Addresses and counts that are out of range.
params, mem = build([".J"])
mem.put(THROW_INFO, struct.pack("<iIII", 0, 0, 0, 0xFFFFFFF0))  # the array RVA points where nothing is mapped
check("an RVA beyond the readable range", decode(params, mem)["status"] == "unreadable")
check("an RVA that overflows 64 bits", m.decode_throw(CODE, [MAGIC, THROWN, THROW_INFO, (1 << 64) - 8], mem.read)["status"] == "unreadable")
params, mem = build([".J"], array_rva=0)
check("a zero array RVA", decode(params, mem)["status"] == "bad_throw_info")
params, mem = build([".J"], count=-1)
check("a negative type count", decode(params, mem)["status"] == "bad_throw_info")
params, mem = build([".J"] * 17, value=None)
check("17 catchable types is over the cap", decode(params, mem)["status"] == "too_many_types")
params, mem = build([".J"] * 16)
check("16 catchable types is the cap", decode(params, mem)["status"] == "ok" and decode(params, mem)["type_count"] == 16)
params, mem = build([".J"] * 3)
check("a lower cap is honoured", m.decode_throw(CODE, params, mem.read, max_types=2)["status"] == "too_many_types")
params, mem = build([".J"], count=2)
check("a count larger than the stored array reads past it", decode(params, mem)["status"] in ("unreadable", "bad_throw_info"))
params, mem = build([".J"])
mem.put(BASE + ARRAY_RVA + 4, struct.pack("<I", 0))
check("a zero catchable type RVA", decode(params, mem)["status"] == "bad_throw_info")
params, mem = build([".J"])
mem.put(BASE + 0x2200 + 4, struct.pack("<I", 0))
check("a zero type_info RVA", decode(params, mem)["status"] == "bad_throw_info")

# 6. Types that are not a scalar, and names that are not reported.
params, mem = build([".?AVruntime_error@std@@", ".?AVexception@std@@"], sizes=[24, 16])
record = decode(params, mem)
check("a std::runtime_error has no value", record["status"] == "ok" and "value" not in record)
check("its types are named", [t["c_type"] for t in record["types"]] == ["std::runtime_error", "std::exception"])
params, mem = build([".?AV_com_error@@"], sizes=[24])
check("a _com_error is named and has no value", decode(params, mem)["types"][0]["c_type"] == "_com_error" and "value" not in decode(params, mem))
params, mem = build([".?AVSecretTitleThing@@", ".?AVBase@Acme@@"], sizes=[40, 8])
record = decode(params, mem)
check("a class of the title is 'other'", [t["type"] for t in record["types"]] == ["other", "other"])
check("and its name is nowhere in the record", "Secret" not in json.dumps(record) and "Acme" not in json.dumps(record))
params, mem = build([".J"], sizes=[8])
check("a long of the wrong size has no value", "value" not in decode(params, mem))
params, mem = build([".M"], sizes=[4])
check("a float is named and has no value", decode(params, mem)["types"][0]["c_type"] == "float" and "value" not in decode(params, mem))
params, mem = build([".J", ".?AVSomething@@"], sizes=[4, 4])
check("only the first catchable type decides the value", decode(params, mem)["value"]["hresult"] == "E_INVALIDARG")
params, mem = build([".?AVSomething@@", ".J"], sizes=[4, 4])
check("a class first means no value", "value" not in decode(params, mem))

# 7. Names that are not a decorated type name.
for label, name in (("no leading dot", "J"), ("a path", ".C:\\dir\\file"), ("a space", ".J x"), ("empty name", ".")):
    params, mem = build([name])
    check(f"a type name with {label} is unreadable", decode(params, mem)["status"] == "unreadable")
params, mem = build([".J"], name_terminated=False)
mem.put(BASE + 0x3000 + 16, b"." + b"J" * 200)
check("a name that never ends is unreadable", decode(params, mem)["status"] == "unreadable")
params, mem = build([".J"])
mem.put(BASE + 0x3000 + 16, b".\xff\0")
check("a non-ASCII name is unreadable", decode(params, mem)["status"] == "unreadable")
# The name ends just before unmapped memory: it is still read.
params, mem = build([".J"])
mem.bytes = {a: v for a, v in mem.bytes.items() if not (BASE + 0x3000 + 16 + 3 <= a < BASE + 0x3000 + 0x80)}
check("a name that ends before unreadable memory", decode(params, mem)["status"] == "ok")

# 8. Nothing that identifies the run leaves the decoder: no address, no path, no number over 32 bits.
records = []
for hresult in (0x80070057, 0x887A0005, 0x80041234, 0x12345678):
    params, mem = build([".J", ".?AVSecret@@"], long_value(hresult), sizes=[4, 4])
    records.append(decode(params, mem))
records.append(decode(*build([".?AVruntime_error@std@@"], sizes=[24])))


def numbers(item):
    if isinstance(item, dict):
        for value in item.values():
            yield from numbers(value)
    elif isinstance(item, list):
        for value in item:
            yield from numbers(value)
    elif isinstance(item, int) and not isinstance(item, bool):
        yield item


for record in records:
    text = json.dumps(record) + m.describe(record)
    check("no number over 32 bits in " + m.describe(record), all(abs(n) < (1 << 32) for n in numbers(record)))
    check("no address, path or long hex in " + m.describe(record),
          not re.search(r"0x[0-9a-fA-F]{9,}|[A-Za-z]:\\|[/\\]|7ff|180000", text), text)

# 9. The one-line form.
check("describe: a named HRESULT",
      m.describe(decode(*build([".J"], long_value(0x887A0005)))) == "msvc-throw ok types=.J count=1 long=DXGI_ERROR_DEVICE_REMOVED category=dxgi")
check("describe: a refusal", m.describe({"status": "bad_magic"}) == "msvc-throw bad_magic")
check("describe: a class", m.describe(decode(*build([".?AVruntime_error@std@@"], sizes=[24]))) == "msvc-throw ok types=.?AVruntime_error@std@@ count=1")

# 10. The HRESULT helpers.
check("hresult_name wraps to 32 bits", m.hresult_name(-2147024809) == "E_INVALIDARG")
check("hresult_category", [m.hresult_category(v) for v in (0, 0x80070005, 0x887A0005, 0x887E0001, 0x80004005)]
      == ["success", "win32", "dxgi", "d3d12", "other"])

if failures:
    print("FAIL")
    for label in failures:
        print("  " + label)
    sys.exit(1)
print("PASS")
