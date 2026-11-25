"""Explicit-gradient image_sample_d in tools/startup/gfx2hlsl.py.

  (no arguments)     translation controls, pure Python
  binary <llvm-mc>   reassemble every fixture source and compare it with the recorded words

The address dwords of image_sample_d are ddx, then ddy (one register per coordinate axis each), then the coordinates.
The lowering must pass those registers to SampleGrad in that order: a 2D sample has six, a 2D array seven (the slice
last) and a volume nine. image_sample_c_d would need SampleCmpGrad (shader model 6.8) and is refused by name instead
of being sampled without its gradients.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_sample_l_check import POST, PRE  # noqa: E402
from startup_sampler_heap_check import VS_WORDS, assemble, root_blob  # noqa: E402

TAIL = 's[16:23], s[24:27] dmask:0xf dim:'
IMAGE = {
    'd2': 'image_sample_d v[4:7], v[0:7], ' + TAIL + 'SQ_RSRC_IMG_2D',
    'd2nsa': 'image_sample_d v[4:7], [v0, v1, v2, v3, v1, v0], ' + TAIL + 'SQ_RSRC_IMG_2D',
    'd3': 'image_sample_d v[4:7], v[0:15], ' + TAIL + 'SQ_RSRC_IMG_3D',
    'd2a': 'image_sample_d v[4:7], v[0:7], ' + TAIL + 'SQ_RSRC_IMG_2D_ARRAY',
    'd1': 'image_sample_d v[4:7], [v0, v1, v2], ' + TAIL + 'SQ_RSRC_IMG_1D',
    'cd': 'image_sample_c_d v[4:7], v[0:7], ' + TAIL + 'SQ_RSRC_IMG_2D',
    'do': 'image_sample_d_o v[4:7], v[0:7], ' + TAIL + 'SQ_RSRC_IMG_2D',
    'dtfe': 'image_sample_d v[4:5], v[0:7], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D tfe',
    'dglc': 'image_sample_d v[4:7], v[0:7], ' + TAIL + 'SQ_RSRC_IMG_2D glc',
}
WORDS = {
    'd2': bytes.fromhex('04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e0000803ef202047ef402067e7fc08cbf080f88f00004c400703f8cbf0f1800f804050607000081bf'),
    'd2nsa': bytes.fromhex('04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e0000803ef202047ef402067e7fc08cbf0c0f88f00004c4000102030100000000703f8cbf0f1800f804050607000081bf'),
    'd3': bytes.fromhex('04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e0000803ef202047ef402067e7fc08cbf100f88f00004c400703f8cbf0f1800f804050607000081bf'),
    'd2a': bytes.fromhex('04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e0000803ef202047ef402067e7fc08cbf280f88f00004c400703f8cbf0f1800f804050607000081bf'),
    'd1': bytes.fromhex('04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e0000803ef202047ef402067e7fc08cbf020f88f00004c40001020000703f8cbf0f1800f804050607000081bf'),
    'cd': bytes.fromhex('04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e0000803ef202047ef402067e7fc08cbf080fa8f00004c400703f8cbf0f1800f804050607000081bf'),
    'do': bytes.fromhex('04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e0000803ef202047ef402067e7fc08cbf080fc8f00004c400703f8cbf0f1800f804050607000081bf'),
    'dtfe': bytes.fromhex('04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e0000803ef202047ef402067e7fc08cbf080189f00004c400703f8cbf0f1800f804050607000081bf'),
    'dglc': bytes.fromhex('04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e0000803ef202047ef402067e7fc08cbf082f88f00004c400703f8cbf0f1800f804050607000081bf'),
}


def stage(name):
    return G.Pipeline(VS_WORDS, WORDS[name], root_blob(flags=0)).hlsl_ps()


def grad_call(name):
    """(location, ddx, ddy) register lists of the SampleGrad call."""
    call = re.search(r'SampleGrad\(([^;]*)\);', stage(name))
    assert call, name
    parts = call.group(1).split('float')[1:]
    assert len(parts) == 3, call.group(1)
    return [re.findall(r'F\((v\d+)\)', part) for part in parts]


def refused(name, text):
    try:
        stage(name)
    except G.Unsupported as error:
        assert text in str(error), (name, str(error))
    else:
        raise AssertionError('%s was accepted' % name)


def check_address_count():
    """A count that does not match the dimension is refused before any register is read."""
    for coords, count in (('[v0, v1, v2, v3, v1]', 5), ('[v0, v1, v2, v3, v1, v0, v2]', 7)):
        bare = G.Stage.__new__(G.Stage)
        try:
            bare.image(['v5', coords, 's0'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0x1'], 'd')
        except G.Unsupported as error:
            assert str(error) == 'image_sample_d with %d address registers' % count, (coords, error)
        else:
            raise AssertionError('image_sample_d accepted %d address registers' % count)


def check_lod_cap():
    """A guest sampler capped to MAX_LOD 0 keeps its cap: the static sampler carries it, so SampleGrad cannot reach mip 2.

    The cap fixture is the 2D program with `s_mov_b32 s26, 0` (80039abe) widened to the literal 0x04000000, mip filter
    point (word 2 bits 27:26 = 1), words (0, 0, 0x04000000, 0). A mip filter of 0 keeps the ordinary sampler."""
    words = WORDS['d2']
    assert words.count(bytes.fromhex('80039abe')) == 1
    capped = words.replace(bytes.fromhex('80039abe'), bytes.fromhex('ff039abe') + (0x04000000).to_bytes(4, 'little'))
    pipe = G.Pipeline(VS_WORDS, capped, root_blob(flags=0))
    assert 'SampleGrad(' in pipe.hlsl_ps()
    lines = pipe.manifest().splitlines()
    assert 'smpl 0 0x0 0x0 0x4000000 0x0' in lines and 'smp ' not in pipe.manifest(), pipe.manifest()
    plain = G.Pipeline(VS_WORDS, words, root_blob(flags=0))
    plain.hlsl_ps()
    assert 'smp 0 0x0 0x0 0x0 0x0' in plain.manifest().splitlines() and 'smpl' not in plain.manifest(), plain.manifest()
    print('a capped constant sampler is carried to the static sampler; mip filter 0 keeps the ordinary one')


def controls():
    check_address_count()
    check_lod_cap()
    assert grad_call('d2') == [['v4', 'v5'], ['v0', 'v1'], ['v2', 'v3']], grad_call('d2')
    assert grad_call('d2nsa') == [['v1', 'v0'], ['v0', 'v1'], ['v2', 'v3']], grad_call('d2nsa')
    assert grad_call('d3') == [['v6', 'v7', 'v8'], ['v0', 'v1', 'v2'], ['v3', 'v4', 'v5']], grad_call('d3')
    assert grad_call('d2a') == [['v4', 'v5', 'v6'], ['v0', 'v1'], ['v2', 'v3']], grad_call('d2a')
    assert 'Texture2DArray' in stage('d2a') and 'Texture3D' in stage('d3')
    print('2D, NSA 2D, 3D and 2D array pass ddx, ddy and the coordinates to SampleGrad in address order')

    refused('cd', 'image_sample_c_d')
    refused('do', 'image_sample_d_o')
    refused('d1', 'unverified 1D image sample address or modifiers')
    refused('dtfe', 'image_sample_d modifier tfe')
    print('refused: compare, offset, 1D, tfe')

    assert grad_call('dglc') == grad_call('d2')  # cache-policy bits do not change the sampled value
    print('glc is accepted')


def check_binary(llvm_mc):
    # The assembler takes the wide tuple spelling (8 VGPRs, 16 for a volume); the decoder lists the active ones.
    for name, line in IMAGE.items():
        built = assemble(llvm_mc, PRE + line + '\n' + POST)
        if built != WORDS[name]:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), WORDS[name].hex()))
    print('fixture words match llvm-mc')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif sys.argv[1] == 'binary' and len(sys.argv) == 3:
        check_binary(sys.argv[2])
    else:
        sys.exit(__doc__)
