"""Native/AOT differential checks and rejection controls for BT, BTS, BTR and BTC in cpu_switch_aot.py."""

import ctypes
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_switch_aot import translate
from direct_check import GUARDS, MASK, SENTINELS, input_pairs, observe, register_probe

OPERATIONS = ("bt", "bts", "btr", "btc")
BITS = (16, 32, 64)
IMMEDIATES = {16: (0, 15, 21), 32: (0, 31, 45), 64: (0, 63, 77)}
MEMORY_IMMEDIATES = {16: 19, 32: 39, 64: 70}
NAMES = ([f"{op}{bits}_{form}" for form in ("reg", "imm", "mem") for op in OPERATIONS for bits in BITS]
         + ["jcc_bt", "jcc_btr_nc", "cmov_bt"])
WIDTHS = {name: 8 for name in NAMES}

REJECTIONS = (
    ("neg_bt_sete", "only CF is defined after BT"),
    ("neg_btc_cmovs", "only CF is defined after BTC"),
    ("neg_bt_seto", "only CF is defined after BT"),
    ("neg_bt_setp", "only CF is defined after BT"),
    ("neg_bts_cmovl", "only CF is defined after BTS"),
    ("neg_bt_jbe", "only CF is defined after BT"),
    ("neg_lock_bts", "unsupported prefix"),
    ("neg_bt_undef", "undefined register rax"),
)


def bit_test(operation, bits, value, offset):
    """CF and the written value: the offset is taken modulo the operand width (Intel SDM, BT)."""
    mask = (1 << bits) - 1
    value &= mask
    offset %= bits
    carry = (value >> offset) & 1
    if operation == "bts":
        value |= 1 << offset
    elif operation == "btr":
        value &= ~(1 << offset) & mask
    elif operation == "btc":
        value ^= 1 << offset
    return carry, value


def merge(register, value, bits):
    if bits >= 32:
        return value & ((1 << bits) - 1)  # a 32-bit write zero-extends
    return (register & ~0xffff | value) & MASK


def setcc(*values):
    data = bytearray(b"\xff" * 8)
    for number, value in enumerate(values):
        data[number] = value
    return int.from_bytes(data, "little")


def reference(name, x, y):
    if name in ("jcc_bt", "jcc_btr_nc", "cmov_bt"):
        if name == "jcc_bt":
            carry, _ = bit_test("bt", 32, y, x)
            rax = y ^ MASK if carry else y
        elif name == "jcc_btr_nc":
            carry, value = bit_test("btr", 64, y, x)
            rax = value if carry else (-value) & MASK
        else:
            carry, rdx = bit_test("bts", 64, y, 5)
            first = rdx if carry else x
            carry, value = bit_test("btc", 32, x, rdx)
            second = value if carry else rdx & 0xffffffff
            return rdx, (GUARDS[0], first, second, MASK, GUARDS[1])
        return rax, (GUARDS[0], MASK, MASK, MASK, GUARDS[1])
    operation, form = name.split("_")
    bits = int(operation.lstrip("btsrc"))
    operation = operation[: -len(str(bits))]
    if form == "reg":
        carry, value = bit_test(operation, bits, y, x)
        register = merge(y, value, bits) if operation != "bt" else y
        return register, (GUARDS[0], setcc(carry, 1 - carry), MASK, register, GUARDS[1])
    if form == "imm":
        register, flags = y, []
        for offset in IMMEDIATES[bits]:
            carry, value = bit_test(operation, bits, register, offset)
            flags.append(carry)
            if operation != "bt":
                register = merge(register, value, bits)
        return register, (GUARDS[0], setcc(*flags), MASK, register, GUARDS[1])
    carry, value = bit_test(operation, bits, y, MEMORY_IMMEDIATES[bits])
    stored = (y & ~((1 << bits) - 1) | value) & MASK
    return x, (GUARDS[0], setcc(carry, 1 - carry), MASK, stored, GUARDS[1])


def bit_inputs():
    values = (0, 1, 0x8000, 0xffff, 0x7fffffff, 0x80000000, 0xffffffff, 0x8000000000000000, MASK,
              0x123456789abcdef0, 0xdeadbeefcafebabe, 0x5555555555555555, 0xaaaaaaaaaaaaaaaa)
    offsets = (0, 1, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 0x7f, 0x80, 0xff, 0xffff, 0x10000, 0x7fffffff,
               0x80000000, 0xffffffff, 0x100000020, 0x8000000000000000, MASK)
    pairs = [(offset, value) for offset in offsets for value in values]
    rng = random.Random(0x42544553)
    pairs.extend((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(256))
    return pairs + input_pairs()


def check_generated(original_path):
    source, functions = translate(original_path, NAMES)
    widths = {name: fn.return_width for name, fn in zip(NAMES, functions)}
    if widths != WIDTHS:
        raise AssertionError(f"return widths {widths} != {WIDTHS}")
    for name in (f"{op}{bits}_mem" for op in OPERATIONS for bits in BITS):
        body = source.split(f"lifted_{name}(", 1)[1].split("\n}", 1)[0]
        # One volatile read of the bit base. The MOV and the two SETcc store; BT stores nothing back.
        stores = 3 + (not name.startswith(("bt1", "bt3", "bt6")))
        if body.count("load_always<") != 1 or body.count("store<") != stores:
            raise AssertionError(f"unexpected guest accesses in {name}:\n{body}")


def check_rejections(original_path):
    for name, reason in REJECTIONS:
        try:
            translate(original_path, [name])
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{name}: expected {reason!r}, got {exc}") from exc
            print(f"rejected {name}: {exc}")
        else:
            raise AssertionError(f"accepted negative control {name}")
    print(f"{len(REJECTIONS)} rejection controls passed")


def check(probe_path, original_path, lifted_path):
    check_generated(original_path)
    probe = register_probe(probe_path)
    native, lifted = ctypes.CDLL(original_path), ctypes.CDLL(lifted_path)
    pairs = bit_inputs()
    total = 0
    for name in NAMES:
        source, translation = getattr(native, name), getattr(lifted, "lifted_" + name)
        for x, y in pairs:
            want, trace = reference(name, x, y)
            original, actual = observe(probe, source, x, y), observe(probe, translation, x, y)
            if original != (want, trace, SENTINELS):
                raise AssertionError(f"original {name}({x:#x}, {y:#x}): {original} != {want:#x} {trace}")
            if actual != original:
                raise AssertionError(f"lifted_{name}({x:#x}, {y:#x}): {actual} != {original}")
            total += 1
        print(f"{name}: {len(pairs)} inputs match native and reference")
    print(f"{total} differential comparisons passed for BT, BTS, BTR and BTC at 16/32/64 bits")
    check_rejections(original_path)


if __name__ == "__main__":
    check(*sys.argv[1:])
