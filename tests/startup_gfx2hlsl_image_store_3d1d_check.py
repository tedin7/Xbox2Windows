"""Check full compute image stores to 1D and 3D texture UAVs."""
import argparse

import startup_compute_image_store_check as A
from startup_gfx2hlsl_compute_check import G, assemble, refused, root_blob
from startup_pixel_image_store_check import WORDS as PIXEL_WORDS, translate as pixel_pipeline


FORMS = {
    'volume': ('3D', 'v[2:4]', 's[16:23]', '101f20f002080400', 'uint3(v2, v3, v4)'),
    'volume-nsa': ('3D', '[v2, v3, v5]', 's[16:23]', '121f20f00208040003050000', 'uint3(v2, v3, v5)'),
    'line': ('1D', 'v0', 's[8:15]', '001f20f000080200', 'uint(v0)'),
}
VALUES = A.ARRAY_VALUES
DATA = bytes.fromhex(
    'ff02107e0000403f'  # v_mov_b32_e32 v8, 0x3f400000
    'ff02127e0000803e'  # v_mov_b32_e32 v9, 0x3e800000
    'ff02147e00008140'  # v_mov_b32_e32 v10, 0x40810000
    'ff02167e010081bf'  # v_mov_b32_e32 v11, 0xbf810001
)


def group(form):
    return (4, 1, 1) if form == 'line' else A.fixture.GROUP


def program(form, store=None):
    # s[12:13] is the fixed table pointer. Compute the global X before the 1D descriptor overwrites s10.
    prefix = bytes.fromhex('07038cbe80038dbe')  # s_mov_b32 s12, s7; s_mov_b32 s13, 0
    if form == 'line':
        address = bytes.fromhex(
            '000046d70a040104'  # v_lshl_add_u32 v0, s10, 2, v0
            '06020cf4400000fa'  # s_load_dwordx8 s[8:15], s[12:13], 0x40
        )
    else:
        address = bytes.fromhex(
            '06040cf4400000fa'  # s_load_dwordx8 s[16:23], s[12:13], 0x40
            '020046d70a040104'  # v_lshl_add_u32 v2, s10, 2, v0
            '030046d70b020504'  # v_lshl_add_u32 v3, s11, 1, v1
        )
        address += bytes.fromhex('81020a7e' if form == 'volume-nsa' else '8102087e')  # z = 1
    instruction = bytes.fromhex(FORMS[form][3]) if store is None else store
    return prefix + address + bytes.fromhex('7fc08cbf') + DATA + instruction + A.ARRAY_END


def pipeline(form, store=None):
    return G.ComputePipeline(program(form, store), root_blob(3), A.RSRC2, group(form))


def controls(llvm_mc):
    ps = pixel_pipeline(PIXEL_WORDS).ps
    for form, (dim, address, descriptor, words, index) in FORMS.items():
        dim_mod = f'dim:SQ_RSRC_IMG_{dim}'
        asm = f'image_store v[8:11], {address}, {descriptor} dmask:0xf {dim_mod} unorm'
        store = bytes.fromhex(words)
        assert G.D.decode_instruction(store, 0)[2] == asm
        if llvm_mc:
            assert assemble(llvm_mc, asm + '\n') == store, asm
        pipe = pipeline(form)
        hlsl, manifest = pipe.hlsl_cs(), pipe.manifest()
        # Manual mutants: dropping the third coordinate or mapping 1D to 2D must fail these assertions.
        assert f'RWTexture{dim}<float4> w0 : register(u0);' in hlsl, hlsl
        assert f'w0[{index}] = asfloat(uint4(v8, v9, v10, v11));' in hlsl, hlsl
        assert f'uav 0 tex{dim.lower()} float table 1 64 channels 4' in manifest.splitlines(), manifest
        mods = ['dmask:0xf', dim_mod, 'unorm']
        refused('only 2D images', lambda: ps.op_image_store(['v[8:11]', address, descriptor], mods))
        wrong_width = 'v[0:1]'
        refused('with 2 address registers', lambda: pipe.cs.op_image_store(['v[8:11]', wrong_width, descriptor], mods))
        for d16 in (False, True):
            partial = bytearray(store)
            partial[1] = (partial[1] & 0xf0) | 0x3
            if d16:
                partial[7] |= 0x80
            # A compute store has no partial typed-UAV write: the channels outside DMASK are written as zero.
            partial_hlsl = pipeline(form, bytes(partial)).hlsl_cs()
            assert f'w0[{index}] = ' in partial_hlsl and '0u' in partial_hlsl.split(f'w0[{index}] = ')[1].split(';')[0] \
                or d16, partial_hlsl
        # d16 with the full mask takes the same packed-half path as 2D: two data VGPRs unpacked low half first.
        # Manual mutant: dropping the f16tof32 unpack for these kinds must fail this assertion.
        packed = bytearray(store)
        packed[7] |= 0x80
        d16_hlsl = pipeline(form, bytes(packed)).hlsl_cs()
        d16_store = f'w0[{index}] = float4(f16tof32(v8 & 0xffffu), f16tof32(v8 >> 16), f16tof32(v9 & 0xffffu), f16tof32(v9 >> 16));'
        assert d16_store in d16_hlsl, d16_hlsl
    print('Compute 1D, 3D and 3D NSA stores, d16 full-mask stores, manifest and refusal checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    controls(parser.parse_args().llvm_mc)
