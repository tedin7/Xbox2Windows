"""A slot-86 pixel shader with a kill block after its first s_endpgm translates and draws, without an LLVM library.

Source-built gfx1013 code only. The pixel shader is the one of startup_stream_interface_check (red follows TEXCOORD u),
with a leading `s_cbranch_execz` to a second exit, `exp null off, off, off, off done vm; s_endpgm`, placed after the
program's first s_endpgm (the layout LLVM and Mesa give a kill or demote). Every lane of the test draw is live, so the
branch is never taken and the picture is the one of the one-exit shader.

Without arguments only the translator cases run:
  two exits:  the request translates (status ok) and the null export is dropped.
  controls:   without the translator's decoder the code is refused as before (an unverified multi-exit program); a body
              instruction the two-exit rule does not allow, a block no branch enters, a second exit nothing jumps over
              and a vertex stage with two exits are all refused with their own reason.
With a consumer and a backend they run first, then the draw on an 8x4 target (left half drawn, same expected gradient).
"""
import os
from pathlib import Path
import struct
import sys
import tempfile

import startup_stream_interface_check as S
from startup_stream_interface_check import gfx2hlsl
from pixel_aot import container_code  # noqa: E402
from resource_inventory import parts  # noqa: E402

EXECZ_TO_KILL = 0xbf880007                      # s_cbranch_execz: target = next word + 7 words = the kill block
NULL_EXPORT = (0xf8001890, 0x00000000)          # exp null off, off, off, off done vm
END = 0xbf810000


def two_exit_words(entry=EXECZ_TO_KILL, body=None, kill=NULL_EXPORT):
    body = S.pixel_words(0) if body is None else body
    return [entry, *body, *kill, END]


def two_exit_pixel(words=None, **kwargs):
    words = two_exit_words(**kwargs) if words is None else words
    table = {0x138: b"".join(bytes(r) for r in S.PS_RECORDS), 0x238: struct.pack("<I", len(S.PS_RECORDS))}
    return S.container(0, words, S.PS_BASE, [S.signature(b"ISG1", (S.POSITION, S.COLOR, S.TEXCOORD))], table)


def native_code(container, decoder, stage=0):
    return container_code(parts(container)[b"DXIL"], stage=stage, decoder=decoder)


def refused(why, container, decoder, stage=0):
    try:
        native_code(container, decoder, stage)
    except ValueError as error:
        assert why in str(error), (why, str(error))
        return
    raise AssertionError("not refused: " + why)


def translator_cases(directory):
    pixel = two_exit_pixel()
    status, hlsl = S.translate(directory / "two_exits", S.vertex(), pixel)
    assert status == "ok", status
    assert "U(i.param1.x)" in hlsl, hlsl                    # the body is translated as in the one-exit shader
    print("translate: a pixel shader with a kill block after its first s_endpgm translates")
    decode = gfx2hlsl.native_instructions
    refused("Unverified multi-exit program", pixel, None)
    print("control: without the translator's decoder the two exits are still unverified")
    refused("Unverified multi-exit control flow: s_sendmsg",
            two_exit_pixel(body=[0xbf900003, *S.pixel_words(0)]), decode)
    print("control: an instruction outside the two-exit rule is refused")
    refused("Unreachable code or unverified alternate exit",
            two_exit_pixel(entry=0xbf880006, body=S.pixel_words(0), kill=(*NULL_EXPORT, 0xbf800000)), decode)
    print("control: a branch that does not land after the first exit is refused")
    refused("Unreachable code or unverified alternate exit", two_exit_pixel(entry=0xbf800000), decode)
    print("control: a second exit nothing branches over is refused")
    refused("Expected termination followed by code padding", two_exit_pixel(words=two_exit_words() + [0xbf800000]), decode)
    print("control: code after the last s_endpgm is refused")
    vertex = S.container(1, S.vertex_words() + [0xbf810000], S.VS_BASE, [], {0x29c: struct.pack("<I", 0)})
    refused("Unverified multi-exit program", vertex, decode, stage=1)
    print("control: a vertex stage with two exits is refused")
    return 0


S_MOV_EXEC_ZERO = 0xbefe0480                    # s_mov_b64 exec, 0: every lane is killed before the branch


def kill_path_draw(directory, consumer, backend):
    """The discard path: exec is cleared, s_cbranch_execz takes the kill exit, so the whole target keeps the clear colour."""
    original = S.pixel
    S.pixel = lambda *args, **kwargs: two_exit_pixel(words=[S_MOV_EXEC_ZERO, *two_exit_words()])
    try:
        pixels = S.draw(directory / "kill", consumer, backend, False)
    finally:
        S.pixel = original
    if pixels is None:
        return 77
    for y in range(S.HEIGHT):
        for x in range(S.WIDTH):
            got = tuple(pixels[(y * S.WIDTH + x) * 4:(y * S.WIDTH + x) * 4 + 4])
            assert got == (0, 0, 255, 255), "pixel (%d, %d) is %s: the killed lanes must keep the clear" % (x, y, got)
    print(f"{backend} draw: the kill path leaves every pixel at the clear colour")
    return 0


def draw_cases(directory, consumer, backend):
    original = S.pixel
    S.pixel = lambda *args, **kwargs: two_exit_pixel()
    try:
        result = S.check(directory, consumer, backend)
    finally:
        S.pixel = original
    return result or kill_path_draw(directory, consumer, backend)


if __name__ == "__main__":
    saved = os.environ.get("X2W_FXC")
    os.environ["X2W_FXC"] = ""
    try:
        with tempfile.TemporaryDirectory() as temp:
            result = translator_cases(Path(temp) / "translate")
            if len(sys.argv) > 1:
                os.environ.pop("X2W_FXC", None) if saved is None else os.environ.update(X2W_FXC=saved)
                result = draw_cases(Path(temp), *sys.argv[1:])
    finally:
        if saved is None:
            os.environ.pop("X2W_FXC", None)
        else:
            os.environ["X2W_FXC"] = saved
    sys.exit(result)
