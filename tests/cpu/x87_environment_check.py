"""Check the x87 leaf-function profile's environment stores: FNSTENV and FNSAVE, and their WAIT-prefixed forms
FSTENV and FSAVE, which the profile refuses because WAIT leaves the status word unknown.

The translator stores the environment only where the status word is known, which is right after FNINIT, so the
stored words are exact. This check translates short functions through the checked entry (Checked) and compares
the bytes that the emitted stores write with the 32-bit protected-mode layouts of the Intel SDM, which it builds
itself: control word, status word, tag word, the four pointer words, and for FNSAVE ST(0) to ST(7) as 80-bit
values. It also checks what the entry must refuse. The bytes of an empty register are not defined by the SDM,
so the profile stores its model's slots there; the check gives each slot its own value, which proves the
register order and the width of each store.

Usage: python tests/cpu/x87_environment_check.py
"""
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from iced_x86 import Decoder  # noqa: E402
from ffx_version_query import lower  # noqa: E402
from x87_lower import Checked  # noqa: E402

REJECT = "{{ x87_status = {status}; return 0; }}"
FNINIT_CONTROL_WORD = 0x037F
FNINIT, WAIT, FNSTENV, FNSAVE = "db e3", "9b", "d9 31", "dd 31"
FLDCW_RESTORE = "d9 2a"  # fldcw [rdx]: the caller passes 0x027F, which a function must restore before it returns
RETURN_ZERO = "31 c0 c3"
STORE = re.compile(r"store<uint(\d+)_t>\(a(?: \+ (\d+))?, ([^;]+)\);")
SLOT = re.compile(r"x87_slot\[(\d)\]\.(sig|se)")
# One distinct model value per slot; slot k is physical register 7 - k, which is ST(7 - k) with an empty stack.
SLOTS = {k: (0x8000000000000000 | (k + 1) << 8 | k, 0x3FF0 + k) for k in range(8)}


def translate(hex_bytes, checked=True):
    body = list(Decoder(64, bytes.fromhex(hex_bytes.replace(" ", "")), ip=0x1000))
    return lower(body, [], {}, {}, {}, x87=Checked(REJECT) if checked else True)


def rejection(hex_bytes, checked=True):
    try:
        translate(hex_bytes, checked)
    except ValueError as exc:
        return str(exc)
    return None


def evaluate(text, control_word):
    if re.fullmatch(r"0x[0-9a-f]+|\d+", text):
        return int(text, 0)
    if text == "x87_cw":
        return control_word
    slot = SLOT.fullmatch(text)
    if slot:
        return SLOTS[int(slot[1])][0 if slot[2] == "sig" else 1]
    raise AssertionError(f"store of an expression the check does not know: {text}")


def written(code, size):
    """The bytes that the function's guest stores write at [rcx], from the stores in its C++."""
    image = bytearray(size)
    touched = bytearray(size)
    for width, offset, text in STORE.findall(code):
        width, offset = int(width) // 8, int(offset or 0)
        assert offset + width <= size, f"store of {width} bytes at {offset} is outside the {size}-byte image"
        assert not any(touched[offset:offset + width]), f"two stores write the bytes at {offset}"
        image[offset:offset + width] = evaluate(text, FNINIT_CONTROL_WORD).to_bytes(width, "little")
        touched[offset:offset + width] = b"\xff" * width
    return bytes(image), bytes(touched)


def environment():
    """The SDM's 28-byte environment after FNINIT: the control word, a clear status word (TOP 0), a tag word
    that marks all eight registers empty, and zero instruction and data pointers and opcode. The reserved
    halfwords at 2, 6, 10 and 26 read 0xFFFF as natively (Intel and a second Windows host measured)."""
    return struct.pack("<7I", 0xFFFF0000 | FNINIT_CONTROL_WORD, 0xFFFF0000, 0xFFFFFFFF, 0, 0, 0, 0xFFFF0000)


def registers():
    return b"".join(struct.pack("<QH", *SLOTS[7 - k]) for k in range(8))  # ST(k) is slot 7 - k


def check_stores(name, program, size, expected, mask):
    image, touched = written(translate(program), size)
    for offset in range(size):
        if mask[offset] and image[offset] != expected[offset]:
            raise AssertionError(f"{name}: byte {offset} is {image[offset]:#04x}, expected {expected[offset]:#04x}")
        if mask[offset] and not touched[offset]:
            raise AssertionError(f"{name}: byte {offset} is not written")


def defined(size):
    return bytes([1]) * size


def check_accepted():
    env, full = environment(), environment() + registers()
    # FINIT is WAIT and FNINIT: the WAIT comes before FNINIT, so the status word is still known at the store.
    for init in (FNINIT, f"{WAIT} {FNINIT}"):
        check_stores("FNSTENV", f"{init} {FNSTENV} {FLDCW_RESTORE} {RETURN_ZERO}", 28, env, defined(28))
        check_stores("FNSAVE", f"{init} {FNSAVE} {FLDCW_RESTORE} {RETURN_ZERO}", 108, full, defined(108))
    # FNSAVE initializes as FNINIT does, so the function must restore the control word, as after FNINIT.
    assert "x87_status = 3" in translate(f"{FNINIT} {FNSAVE} {RETURN_ZERO}"), "no check of the control word"


def check_rejected():
    cases = {
        # The status word is not known at the entry, and the profile has no exception flags to store.
        "FNSTENV without FNINIT": (f"{FNSTENV} {RETURN_ZERO}", True, "not known"),
        "FNSAVE without FNINIT": (f"{FNSAVE} {RETURN_ZERO}", True, "not known"),
        # An x87 instruction between FNINIT and the store may raise an exception flag or move the pointers.
        "FNSTENV after an FLD": (f"{FNINIT} db 29 dd d8 {FNSTENV} {RETURN_ZERO}", True, "not known"),
        # FSTENV and FSAVE are WAIT and the store, and the SDM leaves C0 to C3 undefined after WAIT, as the
        # profile already treats FSTSW (cpu_aot_x87_status): the word the store would write is not known.
        "FSTENV after FNINIT": (f"{FNINIT} {WAIT} {FNSTENV} {RETURN_ZERO}", True, "not known"),
        "FSAVE after FNINIT": (f"{FNINIT} {WAIT} {FNSAVE} {RETURN_ZERO}", True, "not known"),
        # The 16-bit layouts (operand-size prefix) are not stored.
        "16-bit FNSTENV": (f"{FNINIT} 66 {FNSTENV} {RETURN_ZERO}", True, "unsupported x87 operand"),
        "16-bit FNSAVE": (f"{FNINIT} 66 {FNSAVE} {RETURN_ZERO}", True, "unsupported x87 operand"),
        # FNSAVE initializes the unit, which the fixed profile (control word 0x027F) cannot represent.
        "FNSAVE in the fixed profile": (f"{FNSAVE} {RETURN_ZERO}", False, "unsupported x87"),
    }
    for name, (program, checked, text) in cases.items():
        reason = rejection(program, checked)
        assert reason is not None, f"{name}: accepted"
        assert text in reason, f"{name}: rejected for another reason: {reason}"


if __name__ == "__main__":
    check_accepted()
    check_rejected()
    print("x87 environment stores: ok")
