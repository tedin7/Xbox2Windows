"""Synthetic bounded shader exit checks, with optional D3D12 GPU readback."""
from itertools import product
import os
from pathlib import Path
import struct
import sys
import tempfile
from unittest.mock import patch

from startup_buffer_address_check import root_with_one_constant
import startup_stream_interface_check as S
from startup_stream_interface_check import gfx2hlsl
import pixel_aot  # noqa: E402
from pixel_aot import container_code  # noqa: E402
from resource_inventory import parts  # noqa: E402

decode_native = gfx2hlsl.native_instructions
END = 0xbf810000
NOP = 0xbf800000
CLEAR = (0, 0, 255, 255)


def pixel(words):
    table = {0x138: b"".join(bytes(r) for r in S.PS_RECORDS), 0x238: struct.pack("<I", len(S.PS_RECORDS))}
    signature = S.signature(b"ISG1", (S.POSITION, S.COLOR, S.TEXCOORD))
    return S.container(0, words, S.PS_BASE, [signature], table)


def vertex(words):
    table = {0x1dc: b"".join(bytes(r) for r in S.VS_RECORDS), 0x29c: struct.pack("<I", len(S.VS_RECORDS))}
    signature = S.signature(b"OSG1", (S.POSITION, S.COLOR, S.TEXCOORD))
    return S.container(1, words, S.VS_BASE, [signature], table)


def native_code(blob, decoder=gfx2hlsl.native_instructions, stage=0):
    return container_code(parts(blob)[b"DXIL"], stage=stage, decoder=decoder)


def refused(why, blob, decoder=gfx2hlsl.native_instructions, stage=0):
    try:
        native_code(blob, decoder, stage)
    except ValueError as error:
        assert why in str(error), (why, str(error))
        return
    raise AssertionError("expected refusal: " + why)


def branch(opcode, skipped):
    return 0xbf800000 | (opcode << 16) | (skipped & 0xffff)


def compare_zero(value):
    # s_cmp_eq_u32 with inline unsigned operands.
    return 0xbf068000 | (128 + value)


def entry_bit(bit):
    # s_bitcmp1_b32 s8, bit. The root constant supplies both independent input bits.
    return 0xbf0d0008 | ((128 + bit) << 8)


def colour(red, green, blue):
    # Three inline RGB values, implicit alpha from another inline operand.
    values = [S.vop1(1, reg, 242 if enabled else 128)
              for reg, enabled in enumerate((red, green, blue, 1), 12)]
    return [*values, 0xf800180f, 0x0f0e0d0c, END]


def three_exits():
    red = colour(1, 0, 0)
    green = colour(0, 1, 0)
    yellow = colour(1, 1, 0)
    # Branch on the two entry values in order. Each fall-through block exports and exits.
    return [entry_bit(0), branch(5, len(red)), *red,
            entry_bit(1), branch(5, len(green)), *green, *yellow]


def literal_colour():
    # The literal bits represent -1.0078125. Add 2 to get 0.9921875 in red.
    return [S.vop1(1, 12, 255), END, S.vop2(3, 12, 244, 12), S.vop1(1, 13, 128),
            S.vop1(1, 15, 242), 0xf800180f, 0x0f0d0d0c, END]


def vertex_exits():
    position_and_colour = S.vertex_words()[:-3]
    # Both paths export a constant TEXCOORD, which the existing pixel fixture reads as red.
    low = [S.vop1(1, 9, 128), 0xf8000a1f, 0x09090909, END]
    high = [S.vop1(1, 9, 242), 0xf8000a1f, 0x09090909, END]
    return [*position_and_colour, entry_bit(0), branch(5, len(low)), *low, *high]


def no_colour_exit():
    return [entry_bit(0), branch(5, 1), END, *colour(0, 1, 0)]


def shared_exit():
    green = colour(0, 1, 0)
    # The last exit is reached both after an export and by skipping it.
    return [entry_bit(0), branch(5, 1), END, entry_bit(1), branch(4, len(green) - 1), *green]


def branch_tail():
    # One path reaches the colour block via the final unconditional branch.
    green = colour(0, 1, 0)
    return [entry_bit(0), branch(5, len(green)), *green, branch(2, -len(green) - 1)]


def kill_words(clear_exec):
    # A wave that has no live lane skips the colour export and ends via a null export.
    body = S.pixel_words(0)
    return [*([0xbefe0480] if clear_exec else []), branch(8, len(body)), *body, 0xf8001890, 0, END]


def literal_as_exit(code):
    decoded = decode_native(code)
    return sorted([*decoded, (4, "s_endpgm")]) if code[4:8] == struct.pack("<I", END) else decoded


def translated(directory, vs, ps, root=S.EMPTY_ROOT):
    status, _ = S.translate(directory, vs, ps, root)
    assert status == "ok", status


def translator_cases(directory):
    root = root_with_one_constant()
    translated(directory / "pixel", S.vertex(), pixel(three_exits()), root)
    translated(directory / "vertex", vertex(vertex_exits()), S.pixel(), root)
    literal = pixel(literal_colour())
    assert native_code(literal) == struct.pack("<8I", *literal_colour())
    translated(directory / "literal", S.vertex(), literal)
    with patch.object(gfx2hlsl, "native_instructions", literal_as_exit):
        status, _ = S.translate(directory / "literal_mutant", S.vertex(), literal)
    assert status == "unsupported: invalid stream shader: stage 0, 2 decoded exits: unreachable instruction at +0x8", status

    for clear_exec in (False, True):
        translated(directory / f"kill_{clear_exec}", S.vertex(), pixel(kill_words(clear_exec)))
    refused("Unverified multi-exit program", literal, None)
    refused("stage 0, 2 decoded exits: unreachable instruction at +0x8", pixel([END, NOP, END]))
    # An alignment NOP a direct branch skips runs nothing: it may stay unreachable, an unreachable exit may not.
    native_code(pixel([branch(4, 1), NOP, END]))
    refused("stage 0, 1 decoded exits: branch at +0x0 targets outside decoded instruction boundaries (0x44)",
            pixel([branch(4, 16), END]))
    refused("stage 0, 1 decoded exits: branch at +0x0 targets outside decoded instruction boundaries (0x8)",
            pixel([branch(4, 1), *literal_colour()]))
    refused("stage 0, 1 decoded exits: last instruction is neither an exit nor an unconditional branch", pixel([END, NOP]))
    refused("stage 0, 1 decoded exits: computed jump s_setpc_b64 at +0x0", pixel([0xbe802000, END]))
    refused("stage 0, 9 decoded exits: expected 1 to 8 exits", pixel([END] * 9))
    eight = [word for _ in range(7) for word in (branch(4, 1), END)] + [END]
    native_code(pixel(eight))
    compute = S.container(5, eight, S.PS_BASE, [], {})
    native_code(compute, stage=5)
    mesh = S.container(13, eight, S.PS_BASE, [], {})
    refused("stage 13, 8 decoded exits: missing mesh allocation", mesh, stage=13)
    native_code(S.container(13, [0xbf900009, *eight], S.PS_BASE, [], {}), stage=13)
    translated(directory / "branch_tail", S.vertex(), pixel(branch_tail()), root)

    # GETPC reads the address of the following instruction; 24 reaches byte 28.
    getpc = [0xbe901f00, 0x8010ff10, 24, 0x82118011]
    table = [END, END, 7, END]
    image = getpc + table
    assert native_code(vertex(image), stage=1) == struct.pack("<8I", *image)

    def unreadable_tail(code):
        if len(code) > 20:
            raise ValueError("undecodable table")
        return decode_native(code)

    assert native_code(vertex(image), unreadable_tail, stage=1) == struct.pack("<8I", *image)
    with patch.object(pixel_aot, "_vertex_getpc_tail", return_value=False):
        refused("stage 1, 3 decoded exits: unreachable instruction at +0x14", vertex(image), stage=1)
    refused("stage 1, 3 decoded exits: unreachable instruction at +0x14",
            vertex([NOP, NOP, NOP, NOP, *table]), stage=1)
    for displacement in (4, 28):  # points inside code, or just beyond the image
        refused("stage 1, 3 decoded exits: unreachable instruction at +0x14",
                vertex([getpc[0], getpc[1], displacement, getpc[3], *table]), stage=1)
    refused("stage 1, 3 decoded exits: unreachable instruction at +0x18",
            vertex([*getpc, 0xbe921f00, *table]), stage=1)
    refused("stage 1, 2 decoded exits: unreachable instruction at +0x14",
            vertex([getpc[0], getpc[1], 24, getpc[3], END, 0xbe921f00, 7, END]), stage=1)
    # A reachable branch entering the supposed table cannot close the prefix graph.
    refused("stage 1, 2 decoded exits: unreachable instruction at +0x18",
            vertex([*getpc, branch(4, 2), END, 7, END]), stage=1)

    no_position = vertex([compare_zero(0), branch(4, 1), END, *S.vertex_words()])
    status, _ = S.translate(directory / "no_position", no_position, S.pixel())
    assert status == "unsupported: early exit without exports in a vertex stage", status
    translated(directory / "no_colour", S.vertex(), pixel(no_colour_exit()), root)
    translated(directory / "shared_exit", S.vertex(), pixel(shared_exit()), root)
    print("translate: decoded exits, literal mutant, vertex export proof and malformed flow controls passed")


def check_colour(pixels, want):
    assert len(pixels) == S.WIDTH * S.HEIGHT * 4, len(pixels)
    mismatch = {}
    for i in range(S.WIDTH * S.HEIGHT):
        got = tuple(pixels[4 * i:4 * i + 4])
        if got != want:
            mismatch[i % S.WIDTH, i // S.WIDTH] = got
    return mismatch


def draw_cases(directory, consumer, backend):
    def draw(name, vs, ps, full=True, *, entry=None):
        return S.draw(directory / name, consumer, backend, False, vertex_blob=vs,
                      pixel_blob=ps, full_target=full, entry_word=entry)

    for clear_exec in (False, True):
        pixels = draw(f"kill_draw_{clear_exec}", S.vertex(), pixel(kill_words(clear_exec)), False)
        if pixels is None:
            return 77
        mismatch = check_colour(pixels, CLEAR) if clear_exec else S.check_pixels(pixels)
        assert not mismatch, mismatch

    for first, second in product(range(2), repeat=2):
        expected = (255, 0, 0, 255) if not first else (0, 255, 0, 255) if not second else (255, 255, 0, 255)
        pixels = draw(f"pixel_draw_{first}{second}", S.vertex(), pixel(three_exits()), entry=first | second << 1)
        assert pixels is not None
        assert not (mismatch := check_colour(pixels, expected)), mismatch

    pixels = draw("literal_draw", S.vertex(), pixel(literal_colour()))
    red = round((2 + struct.unpack("<f", struct.pack("<I", END))[0]) * 255)
    assert not (mismatch := check_colour(pixels, (red, 0, 0, 255))), mismatch
    for select in range(2):
        pixels = draw(f"vertex_draw_{select}", vertex(vertex_exits()), S.pixel(), entry=select)
        assert not (mismatch := check_colour(pixels, (255 * select, 0, 0, 255))), mismatch

    for select in range(2):
        pixels = draw(f"no_colour_{select}", S.vertex(), pixel(no_colour_exit()), entry=select)
        assert not (mismatch := check_colour(pixels, (0, 255, 0, 255) if select else CLEAR)), mismatch

    for entry in (1, 3):
        pixels = draw(f"shared_exit_{entry}", S.vertex(), pixel(shared_exit()), entry=entry)
        assert not (mismatch := check_colour(pixels, (0, 255, 0, 255) if entry == 3 else CLEAR)), mismatch

    pixels = draw("branch_tail_draw", S.vertex(), pixel(branch_tail()), entry=1)
    assert not (mismatch := check_colour(pixels, (0, 255, 0, 255))), mismatch

    lower = gfx2hlsl.Stage.lower
    words = three_exits()
    second_exit = 4 * [i for i, word in enumerate(words) if word == END][1]

    def fall_through(stage, opcode, asm, operands, modifiers):
        if stage.kind == "ps" and opcode == "s_endpgm" and stage.pc == second_exit:
            return None
        return lower(stage, opcode, asm, operands, modifiers)

    with patch.object(gfx2hlsl.Stage, "lower", fall_through):
        pixels = draw("fall_through_mutant", S.vertex(), pixel(words), entry=1)
    mismatch = check_colour(pixels, (0, 255, 0, 255))
    assert len(mismatch) == S.WIDTH * S.HEIGHT and set(mismatch.values()) == {(255, 255, 0, 255)}, mismatch
    print(f"{backend}: full-target readbacks match CPU references and the fall-through mutant differs at every pixel")
    return 0


if __name__ == "__main__":
    saved = os.environ.get("X2W_FXC")
    os.environ["X2W_FXC"] = ""
    try:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            translator_cases(directory / "translate")
            if saved is None:
                os.environ.pop("X2W_FXC", None)
            else:
                os.environ["X2W_FXC"] = saved
            if len(sys.argv) > 1:
                sys.exit(draw_cases(directory, *sys.argv[1:]))
    finally:
        if saved is None:
            os.environ.pop("X2W_FXC", None)
        else:
            os.environ["X2W_FXC"] = saved
