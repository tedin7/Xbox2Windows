#!/usr/bin/env python3
"""Control for the reserved/hint NOP space in the shared lowering (tools/ffx_version_query.py, #1476).

The AMD APM documents `0F 19 /r`, `0F 1D /r`, `0F 1E /r` (unprefixed) and `0F 1F /r` as reserved hint NOPs
that take a ModRM but access no memory. CET-instrumented code puts `endbr64`/`endbr32` (`F3 0F 1E FA/FB`) at
indirect-branch targets; both have no register, flag, or memory side effects and execute as NOPs on `x64_zen2`.

What is checked, all on exact byte patterns:

  accepted  - the encoding lowers, and the lifted body is identical to the same program with an
              already-accepted NOP. Identical bodies are the proof that the instruction evaluates nothing:
              no address computation, no load, no register or flag write. Block labels are normalised
              because they are numbered by instruction count, not by effect.
  refused   - encodings in the same space that iced decodes as another mnemonic or unaccepted reserved NOP
              stay refused: CLDEMOTE, RDSSPD/RDSSPQ, 0F 0D, 0F 18, 0F 1A-0F 1C. A widened accepted set would
              show up here.

The memory-form cases include a ModRM whose address would fault if evaluated (a non-canonical pointer in
RAX) and a RIP-relative displacement: identical lifted bodies mean neither can be touched.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from iced_x86 import Decoder
from ffx_version_query import lower

# Exact accepted encodings. ModRM 00 = [rax]; 08 = [rax+8]; C0 = register form; 1F 40 00 = [rax+0] dword.
ACCEPTED = {
    "0f1900": "reserved nop [rax]",
    "0f1908": "reserved nop [rax+8]",
    "0f19c0": "reserved nop register form",
    "0f1d00": "reserved nop 0F 1D [rax]",
    "0f1e00": "reserved nop 0F 1E [rax]",
    "f30f1e00": "reserved nop F3 0F 1E [rax] (same iced code; only reg=1/mod=11 is RDSSP)",
    "0f1f4000": "nop dword [rax+0]",
    "660f1900": "reserved nop 0F 19 [rax], 16-bit operand size",
    "660f1e00": "reserved nop 0F 1E [rax], 16-bit operand size",
    "480f1900": "reserved nop 0F 19 [rax], 64-bit operand size",
    "480f1e00": "reserved nop 0F 1E [rax], 64-bit operand size",
    "f30f1efa": "endbr64",
    "f30f1efb": "endbr32",
}

# Encodings in the same space that iced also decodes as RESERVEDNOP but that are not part of this slice, plus
# the ones that decode as their own mnemonic. Listing them keeps a wider accepted set a visible change.
REFUSED = {
    "0f0dc0": "0F 0D Group P (aliases PREFETCH Exclusive)",
    "0f18f0": "0F 18 /6 (PREFETCHIT1 under CPUID Fn8000_0021_EAX)",
    "0f18f8": "0F 18 /7 (PREFETCHIT0)",
    "0f1a00": "0F 1A",
    "0f1b00": "0F 1B",
    "0f1c08": "0F 1C /1",
    "0f1c00": "cldemote [rax]",
    "f30f1ec8": "rdsspd eax",
    "f3480f1ec8": "rdsspq rax",
}

LABEL = re.compile(r"block_\d+")


def body(hex_bytes: str) -> str:
    """Lower one instruction followed by `ret`, with block labels normalised out of the comparison."""
    return LABEL.sub("block_#", lower(list(Decoder(64, bytes.fromhex(hex_bytes + "c3"))), [], {}, {}, {}))


def main() -> int:
    failures = []

    def check(label, condition, detail=""):
        print(f"{label}: {'ok' if condition else 'MISMATCH'}{(' (' + detail + ')') if detail and not condition else ''}")
        if not condition:
            failures.append(label)

    reference = body("0f1f00")  # an already-accepted NOP: the no-effect reference body
    for encoding, description in ACCEPTED.items():
        try:
            lifted = body(encoding)
        except ValueError as error:
            check(f"accepted {encoding} ({description})", False, str(error))
            continue
        check(f"accepted {encoding} ({description})", True)
        check(f"  {encoding} has no effect on the lifted body", lifted == reference)

    for encoding, description in REFUSED.items():
        try:
            body(encoding)
        except ValueError:
            check(f"refused {encoding} ({description})", True)
            continue
        check(f"refused {encoding} ({description})", False, "lowering accepted it")

    if failures:
        print(f"FAIL: {len(failures)} case(s)")
        return 1
    print(f"PASS: {len(ACCEPTED)} accepted forms lowered to a no-op, {len(REFUSED)} neighbours still refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
