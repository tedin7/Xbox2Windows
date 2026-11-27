"""Plain depth export followed by a valid-mask colour export, with a null kill exit."""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_image_3d_check import G, VS_WORDS, compile_hlsl, root_blob
from startup_image_load_d16_check import encode_mimg
from startup_stream_check import container
from startup_stream_draw_check import vop1, vop2
from pixel_aot import container_code
from resource_inventory import parts


MRTZ, COLOR, NULL = 0xf8000081, 0xf800180f, 0xf8001890
END = 0xbf810000
GUARD = 'if (exec & 1) o.depth = F(v6);'
DISCARD = 'if (!(exec & 1)) discard;'
CMPX_POSITIVE = 0x7c220c80  # v_cmpx_lt_f32_e32 0, v6
VERTEX = """\
float4 main(uint id : SV_VertexID) : SV_Position {
    return float4(id == 1 ? 3 : -1, id == 2 ? 3 : -1, 0.5, 1);
}
"""


def pixel(*, depth=MRTZ, color=COLOR, null=NULL, branch=0xbf880000, between=()):
    # POS_FIXED_PT in v0. Unpack x/y, load one float, and keep positive texels.
    prefix = (0xf40c0404, 0xfa000000, vop2(0x1b, 2, 255, 0), 0xffff,
              vop2(0x16, 3, 144, 0), 0xbf8cc07f)
    load = encode_mimg(0, dmask=1, vaddr=2, vdata=6)
    # The live path exports the loaded depth and opaque green.
    live = (vop1(1, 7, 128), vop1(1, 8, 242), depth, 6, *between,
            color, 0x08070807, END)
    tail = (0xbf8c3f70, CMPX_POSITIVE, branch | len(live), *live, null, 0, END)
    return struct.pack('<6I', *prefix) + load + struct.pack(f'<{len(tail)}I', *tail)


def translate(code):
    return G.Pipeline(VS_WORDS, code, root_blob(), ps_hw_inputs=(0x8000, 0x8000)).hlsl_ps()


def checks():
    code = pixel()
    wrapped = container(0, [word for word, in struct.iter_unpack('<I', code)])
    assert container_code(parts(wrapped)[b'DXIL'], stage=0, decoder=G.native_instructions) == code
    hlsl = translate(code)
    assert GUARD in hlsl and hlsl.count(DISCARD) == 2, hlsl
    assert 'SV_Depth' in hlsl and 'SV_Target0' in hlsl and 'register(t0)' in hlsl, hlsl
    ins = G.decode(code)
    at = next(off for off, _, _, asm in ins if asm.startswith('exp mrtz '))
    color_at = next(off for off, _, _, asm in ins if asm.startswith('exp mrt0 '))
    assert G.mrtz_followed_by_vm(ins, at)
    # A branch elsewhere that can enter the colour export splits the basic block.
    other_entry = [(ins[-1][0] + 4, 4, 's_branch', f's_branch label_0x{color_at:x}')]
    assert not G.mrtz_followed_by_vm(ins + other_entry, at)
    for label, mutant in (
        ('no colour vm', pixel(color=COLOR ^ 0x1000)),
        ('no colour done', pixel(color=COLOR ^ 0x800)),
        ('no null vm', pixel(null=NULL ^ 0x1000)),
        ('live lanes can reach null', pixel(branch=0xbf890000)),
        ('mask write', pixel(between=(CMPX_POSITIVE,))),
        ('branch', pixel(between=(0xbf820000,))),
        ('second MRTZ', pixel(between=(MRTZ, 6))),
        ('compressed depth', pixel(depth=MRTZ | 0x402)),
        ('stencil', pixel(depth=MRTZ | 2)),
        ('sample mask', pixel(depth=MRTZ | 4)),
    ):
        try:
            translate(mutant)
        except G.Unsupported:
            continue
        raise AssertionError(f'{label}: invalid shader translated')
    return hlsl


def draw(hlsl, dxc, renderer):
    # Left half is entirely zero; the right half mixes zeros and two nonzero depths.
    samples = [0.0 if x < 16 or (x + y) % 2 == 0 else (0.25 if y % 2 else 0.625)
               for y in range(32) for x in range(32)]
    depths = [value if value else 0.9375 for value in samples]
    colors = bytes(channel for value in samples
                   for channel in ((0, 255, 0, 255) if value else (0, 0, 255, 255)))
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        fixture = directory / 'pixels'
        fixture.write_bytes(struct.pack('<2048f', *samples, *depths) + colors)
        vertex = compile_hlsl(dxc, 'dxc', VERTEX, 'vs', 'vs_6_0', directory)
        # Return a depth for masked lanes before they can discard, keeping their colour at clear blue.
        branch = 'pc = ((exec & 1) == 0) ? 2 : 1;'
        assert hlsl.count(branch) == 1, hlsl
        depth_leak = hlsl.replace(branch, '''
        if (!(exec & 1) && i.pos.x >= 16) {
            o.depth = F(v6);
            o.mrt0 = float4(0, 0, 1, 1);
            return o;
        }
        ''' + branch)
        null_start = hlsl.index('if (pc == 2)')
        variants = (
            ('correct', hlsl, (0, 0)),
            ('depth_leak', depth_leak, (256, 0)),
            ('null_no_discard', hlsl[:null_start] + hlsl[null_start:].replace(DISCARD, '', 1), (768, 768)),
        )
        for name, source, expected in variants:
            shader = compile_hlsl(dxc, 'dxc', source, name, 'ps_6_0', directory)
            result = subprocess.run([renderer, 'warp', '--depth-mask-color', vertex, shader, fixture],
                                    capture_output=True, text=True)
            print(result.stdout, end='')
            print(result.stderr, end='')
            match = re.search(r'depth mismatches=(\d+) color mismatches=(\d+)', result.stdout)
            assert match and tuple(map(int, match.groups())) == expected, (name, result)
            assert result.returncode == bool(expected[0]), (name, result)
            if 'debug_layer=enabled' in result.stdout:
                assert 'debug_errors=0 debug_warnings=0' in result.stdout, (name, result)


if __name__ == '__main__':
    source = checks()
    print('MRTZ translation and refusal controls passed')
    if len(sys.argv) > 1:
        draw(source, *sys.argv[1:])
