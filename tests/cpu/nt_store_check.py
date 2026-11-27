#!/usr/bin/env python3
"""Control for the legacy non-temporal stores in the shared lowering (tools/ffx_version_query.py).

MOVNTPS, MOVNTPD and MOVNTDQ (`0F 2B /r`, `66 0F 2B /r`, `66 0F E7 /r`) store a whole XMM register; like
MOVAPS they raise #GP(0) when the target is not 16-byte aligned. MOVNTI (`0F C3 /r`) stores a 32- or 64-bit
register with no alignment requirement. None of them changes a flag. The non-temporal hint and the weak
ordering of the store are kept by lowering each one to the matching streaming-store intrinsic.

What is checked, all on exact byte patterns:

  accepted  - the encoding lowers to the streaming intrinsic of its width. The XMM forms test the target
              against 15 and raise the aligned-operand fault before the store; MOVNTI does not.
  refused   - the neighbours that are not part of this slice stay refused: an XMM register above XMM2 (the
              register set MOVAPS has here), the MMX MOVNTQ, the SSE4a scalar MOVNTSS and MOVNTSD, the VEX
              VMOVNTPS/VMOVNTPD/VMOVNTDQ, and the MOVNTDQA load. A widened accepted set would show up here.

The native-versus-lifted comparison is tests/cpu/nt_store_fixture.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from iced_x86 import Decoder
from ffx_version_query import lower

ALIGN = "if (target & 15u) raise_aligned_m128_fault();"

# encoding -> (description, intrinsic the body must call, whether the body checks 16-byte alignment)
ACCEPTED = {
    "0f2b01": ("movntps [rcx], xmm0", "_mm_stream_si128", True),
    "660f2b01": ("movntpd [rcx], xmm0", "_mm_stream_si128", True),
    "660fe701": ("movntdq [rcx], xmm0", "_mm_stream_si128", True),
    "0f2b5110": ("movntps [rcx+16], xmm2", "_mm_stream_si128", True),
    "660fe70424": ("movntdq [rsp], xmm0", "_mm_stream_si128", True),
    "0fc301": ("movnti [rcx], eax", "_mm_stream_si32", False),
    "480fc301": ("movnti [rcx], rax", "_mm_stream_si64", False),
    "4c0fc34103": ("movnti [rcx+3], r8", "_mm_stream_si64", False),
}

REFUSED = {
    "0f2b19": "movntps [rcx], xmm3 (the shared lowering models XMM0-XMM2, as for MOVAPS)",
    "0fe701": "movntq [rcx], mm0 (MMX)",
    "f30f2b01": "movntss [rcx], xmm0 (SSE4a)",
    "f20f2b01": "movntsd [rcx], xmm0 (SSE4a)",
    "c5f82b01": "vmovntps [rcx], xmm0 (VEX)",
    "c5f92b01": "vmovntpd [rcx], xmm0 (VEX)",
    "c5f9e701": "vmovntdq [rcx], xmm0 (VEX)",
    "660f382a01": "movntdqa xmm0, [rcx] (non-temporal load)",
}


def body(hex_bytes: str) -> str:
    return lower(list(Decoder(64, bytes.fromhex(hex_bytes + "c3"))), [], {}, {}, {})


def main() -> int:
    failures = []

    def check(label, condition, detail=""):
        print(f"{label}: {'ok' if condition else 'MISMATCH'}{(' (' + detail + ')') if detail and not condition else ''}")
        if not condition:
            failures.append(label)

    for encoding, (description, intrinsic, aligned) in ACCEPTED.items():
        try:
            lifted = body(encoding)
        except ValueError as error:
            check(f"accepted {encoding} ({description})", False, str(error))
            continue
        check(f"accepted {encoding} ({description})", True)
        check(f"  {encoding} stores with {intrinsic}", lifted.count(intrinsic + "(") == 1)
        check(f"  {encoding} {'checks' if aligned else 'does not check'} 16-byte alignment",
              (ALIGN in lifted) == aligned)
        if aligned:
            check(f"  {encoding} raises the fault before the store", lifted.find(ALIGN) < lifted.find(intrinsic))

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
    print(f"PASS: {len(ACCEPTED)} non-temporal stores lowered, {len(REFUSED)} neighbours still refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
