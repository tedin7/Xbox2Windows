"""Digest of generated HLSL that ignores the shared helper prelude.

Every translated stage starts with the same block of helper functions (ldexp32, minmax16_bits, ...) that
tools/startup/gfx2hlsl.py Pipeline.common() emits before the per-shader `cbuffer entry` line. A control that only
asserts "this unrelated translation did not change" must not break when a helper is added or reworded, so the
digest covers the text from that line on. The helpers themselves are guarded by their own draw/readback checks.
"""
import hashlib

BOUNDARY = 'cbuffer entry : register(b0)'


def body(text):
    """The per-shader part of a generated stage; a text without the boundary is returned whole."""
    if isinstance(text, bytes):
        text = text.decode('utf-8').replace('\r\n', '\n')
    at = text.find(BOUNDARY)
    return text[at:] if at >= 0 else text


def digest(*parts, sep='\0'):
    """sha256 of the bodies of the given HLSL texts (str or bytes), joined by sep."""
    return hashlib.sha256(sep.join(body(p) for p in parts).encode()).hexdigest()
