"""Vertex-stage buffer_store_dword..dwordx4 in tools/startup/gfx2hlsl.py (issue #1723).

usage: startup_gfx2hlsl_buffer_store_check.py

A vertex program that writes to a buffer through a descriptor from a root-table slot used to be refused at
`buffer_store_dwordx3` (84 pairs in the offline census). The translation now declares a RWByteAddressBuffer,
emits `Store*` for the `off` form, and writes a `uav <slot> table <param> <byte offset>` line under an
`x2w-manifest 2` header. Programs without a store keep `x2w-manifest 1`.

Instruction bytes are constructed from the MUBUF field layout the checked-in decoder reads; the root
signature is built field by field. Nothing comes from an archive.
"""
import os
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

MUBUF_LOAD_DWORD = 0x0c
STORE_OPS = {'dword': 0x1c, 'dwordx2': 0x1d, 'dwordx4': 0x1e, 'dwordx3': 0x1f}
failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


def encode_mubuf(op, *, imm=0, offen=0, idxen=0, glc=0, slc=0, vaddr=0, vdata=1, srsrc=32, soffset=128):
    w0 = (0b111000 << 26) | ((op & 0xff) << 18) | ((glc & 1) << 14) | ((idxen & 1) << 13) | ((offen & 1) << 12) | (imm & 0xfff)
    w1 = ((soffset & 0xff) << 24) | ((slc & 1) << 22) | (((srsrc // 4) & 0x1f) << 16) | ((vdata & 0xff) << 8) | (vaddr & 0xff)
    return struct.pack('<II', w0, w1)


def v_mov(vdst, inline):
    """VOP1 v_mov_b32 vdst, <inline constant 0..64 encoded as its operand number>."""
    return struct.pack('<I', 0x7e000000 | (vdst << 17) | (1 << 9) | (128 + inline))


SOP2 = {'s_add_u32': 0x00, 's_max_u32': 0x09, 's_and_b32': 0x0e, 's_or_b32': 0x10, 's_mul_i32': 0x26, 's_bfe_u32': 0x27}


def sop2(op, sdst, ssrc0, ssrc1, literal=None):
    """SOP2 word (operand numbers: SGPR n is n, inline integer n is 128 + n, 255 takes the literal)."""
    word = struct.pack('<I', (0b10 << 30) | (op << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0)
    return word if literal is None else word + struct.pack('<I', literal)


# One descriptor table at s[8:9] with an SRV/UAV range of two entries (the same layout the buffer address check uses).
RTS0 = b''.join((struct.pack('<6I', 2, 1, 24, 0, 0, 0), struct.pack('<3I', 0, 0, 36),
                 struct.pack('<2I', 1, 44), struct.pack('<6I', 0, 2, 0, 0, 0, 0)))
PART = b'RTS0' + struct.pack('<I', len(RTS0)) + RTS0
ROOT = b'DXBC' + bytes(16) + struct.pack('<4I', 1, 36 + len(PART), 1, 36) + PART
LOAD_DESCRIPTOR = struct.pack('<3I', 0xf4080804, 0xfa000010, 0xbf8cc07f)  # s_load_dwordx4 s[32:35], s[8:9], 0x10
EXPORT_POSITION = struct.pack('<4I', 0xbf8c3f70, 0xf80008cf, 0x06060606, 0xbf810000)
PS = struct.pack('<3I', 0xf800180f, 0, 0xbf810000)
DEFINE_V = v_mov(1, 3) + v_mov(2, 5) + v_mov(3, 7) + v_mov(4, 9)


def request(folder, vs, ps=PS):
    key = G.pipeline_key(vs, ps, ROOT)
    for suffix, data in (('a', vs), ('b', ps), ('root', ROOT)):
        (folder / f'req_{key:016x}.{suffix}').write_bytes(data)
    result = G.translate_request(str(folder), key)
    return result, folder / f'{key:016x}'


def fail_reason(name):
    path = name.with_suffix('.fail')
    return path.read_text().strip() if path.exists() else ''


def main():
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'X2W_FXC': ''}):
        folder = Path(temp)

        # 1. dwordx3 in the off form: scalar byte offset plus immediate, through a fixed table slot.
        store = encode_mubuf(STORE_OPS['dwordx3'], imm=8, vdata=1, srsrc=32)
        asm = G.decode(store)
        check('decodes as the off form', asm == [(0, 8, 'buffer_store_dwordx3',
                                                  'buffer_store_dwordx3 v[1:3], off, s[32:35], 0 offset:8')], repr(asm))
        result, name = request(folder, LOAD_DESCRIPTOR + DEFINE_V + store + EXPORT_POSITION)
        check('x3 store translates', result == 'ok', '%s %s' % (result, fail_reason(name)))
        if result == 'ok':
            hlsl = name.with_suffix('.vs.hlsl').read_text()
            manifest = name.with_suffix('.txt').read_text().splitlines()
            check('declares the writable view', 'RWByteAddressBuffer u0 : register(u0);' in hlsl)
            check('stores three words at the address', 'a = 8;' in hlsl and 'u0.Store3(a, uint3(v1, v2, v3));' in hlsl)
            check('manifest is version 2 with the uav line',
                  manifest[0] == 'x2w-manifest 2' and 'uav 0 raw table 0 16' in manifest and not any(l.startswith('srv') for l in manifest),
                  repr(manifest))

        # 2. The other widths use the matching Store call.
        for suffix, call, regs in (('dword', 'u0.Store(a, v1);', 1), ('dwordx2', 'u0.Store2(a, uint2(v1, v2));', 2),
                                   ('dwordx4', 'u0.Store4(a, uint4(v1, v2, v3, v4));', 4)):
            result, name = request(folder, LOAD_DESCRIPTOR + DEFINE_V + encode_mubuf(STORE_OPS[suffix]) + EXPORT_POSITION)
            check('store_%s' % suffix, result == 'ok' and call in name.with_suffix('.vs.hlsl').read_text(),
                  '%s %s' % (result, fail_reason(name)))

        # 3. A scalar byte offset adds to the immediate.
        result, name = request(folder, LOAD_DESCRIPTOR + DEFINE_V + encode_mubuf(STORE_OPS['dwordx3'], imm=4, soffset=132) + EXPORT_POSITION)
        check('scalar offset plus immediate', result == 'ok' and 'a = 8;' in name.with_suffix('.vs.hlsl').read_text(),
              '%s %s' % (result, fail_reason(name)))

        # 3b. An unaligned effective offset, and a descriptor whose base was changed, are refused.
        for label, code in (('unaligned inline offset', encode_mubuf(STORE_OPS['dwordx3'], imm=4, soffset=130)),
                            ('unaligned immediate', encode_mubuf(STORE_OPS['dwordx3'], imm=6))):
            result, name = request(folder, LOAD_DESCRIPTOR + DEFINE_V + code + EXPORT_POSITION)
            check('%s is refused' % label, result.startswith('unsupported') and 'unaligned' in fail_reason(name),
                  '%s %s' % (result, fail_reason(name)))
        add_base = struct.pack('<I', 0x80000000 | (32 << 16) | (144 << 8) | 32)  # s_add_u32 s32, s32, 16
        result, name = request(folder, LOAD_DESCRIPTOR + add_base + DEFINE_V + encode_mubuf(STORE_OPS['dwordx3']) + EXPORT_POSITION)
        check('store through a modified descriptor is refused', result.startswith('unsupported') and 'modified' in fail_reason(name),
              '%s %s' % (result, fail_reason(name)))

        # 3c. The descriptor modifications a compute store accepts (#2320, #2546): the stride clear and the
        # structured-to-raw conversion (stride DW1[29:16] extracted, optionally clamped to 1, NUM_RECORDS in DW2
        # scaled by it, OOB_SELECT DW3[29:28] set to 3, stride cleared). The store is the plain one, so the HLSL and
        # manifest are those of the unmodified descriptor; everything else stays refused.
        plain_store = encode_mubuf(STORE_OPS['dwordx3'], imm=8, soffset=13)
        set_s13 = struct.pack('<I', 0xbe800000 | (13 << 16) | (3 << 8) | 128)  # SOP1 s_mov_b32 s13, 0
        plain, name = request(folder, LOAD_DESCRIPTOR + set_s13 + DEFINE_V + plain_store + EXPORT_POSITION)
        check('plain store with a scalar offset translates', plain == 'ok', '%s %s' % (plain, fail_reason(name)))
        plain_hlsl = name.with_suffix('.vs.hlsl').read_text() if plain == 'ok' else None
        plain_manifest = name.with_suffix('.txt').read_text() if plain == 'ok' else None
        extract = sop2(SOP2['s_bfe_u32'], 12, 33, 255, 0xe0010)
        clamp = sop2(SOP2['s_max_u32'], 12, 12, 129)
        clear = sop2(SOP2['s_and_b32'], 33, 33, 255, 0xc000ffff)
        oob_clear = sop2(SOP2['s_and_b32'], 35, 35, 255, 0xcfffffff)
        scale = sop2(SOP2['s_mul_i32'], 34, 34, 12)
        oob_raw = sop2(SOP2['s_or_b32'], 35, 35, 255, 0x30000000)
        check('conversion decodes as captured', [a for (_, _, _, a) in G.decode(extract + clamp + clear + oob_clear + scale + oob_raw)] == [
            's_bfe_u32 s12, s33, 0xe0010', 's_max_u32 s12, s12, 1', 's_and_b32 s33, s33, 0xc000ffff',
            's_and_b32 s35, s35, 0xcfffffff', 's_mul_i32 s34, s34, s12', 's_or_b32 s35, s35, 0x30000000'])
        for label, mod in (('clamped raw conversion', extract + clamp + clear + oob_clear + scale + oob_raw),
                           ('raw conversion', extract + clear + scale + oob_raw),
                           ('stride clear', clear)):
            result, name = request(folder, LOAD_DESCRIPTOR + set_s13 + mod + DEFINE_V + plain_store + EXPORT_POSITION)
            check('%s translates as the plain store' % label, result == 'ok' and plain_hlsl is not None
                  and name.with_suffix('.vs.hlsl').read_text() == plain_hlsl
                  and name.with_suffix('.txt').read_text() == plain_manifest, '%s %s' % (result, fail_reason(name)))
        # Captured programs set the high word of the 32-bit table pointer to 0 before loading; a second load through
        # the same untouched pointer and high word reads the same words, so its stride is the same value.
        table_hi = struct.pack('<I', 0xbe800000 | (9 << 16) | (3 << 8) | 128)  # SOP1 s_mov_b32 s9, 0
        reload = struct.pack('<3I', 0xf4080904, 0xfa000010, 0xbf8cc07f)  # s_load_dwordx4 s[36:39], s[8:9], 0x10
        result, name = request(folder, table_hi + LOAD_DESCRIPTOR + set_s13 + reload + sop2(SOP2['s_bfe_u32'], 12, 37, 255, 0xe0010)
                               + clear + scale + oob_raw + DEFINE_V + plain_store + EXPORT_POSITION)
        reference, ref_name = request(folder, table_hi + LOAD_DESCRIPTOR + set_s13 + DEFINE_V + plain_store + EXPORT_POSITION)
        check('stride of a second load through the same pointer translates as the plain store', result == reference == 'ok'
              and name.with_suffix('.vs.hlsl').read_text() == ref_name.with_suffix('.vs.hlsl').read_text(),
              '%s %s' % (result, fail_reason(name)))
        # After an add on the table pointer, a second load reads other words under the same table offset.
        add_pointer = sop2(SOP2['s_add_u32'], 8, 8, 255, 32)
        refused_mods = {
            'clamp other than 1': extract + sop2(SOP2['s_max_u32'], 12, 12, 130) + clear + oob_clear + scale + oob_raw,
            'NUM_RECORDS not scaled': extract + clamp + clear + oob_clear + oob_raw,
            'NUM_RECORDS scaled by a constant': clear + sop2(SOP2['s_mul_i32'], 34, 34, 132) + oob_raw,
            'OOB_SELECT 2': extract + clamp + clear + oob_clear + scale + sop2(SOP2['s_or_b32'], 35, 35, 255, 0x20000000),
            'stride not cleared': extract + clamp + oob_clear + scale + oob_raw,
            'another bit field': sop2(SOP2['s_bfe_u32'], 12, 33, 255, 0xe0011) + clear + scale + oob_raw,
            'stride of a load after a pointer add': table_hi + add_pointer + reload + sop2(SOP2['s_bfe_u32'], 12, 37, 255, 0xe0010)
            + clear + scale + oob_raw,
            'NUM_RECORDS of a load after a pointer add': table_hi + add_pointer + reload + extract + clear
            + sop2(SOP2['s_mul_i32'], 34, 38, 12) + oob_raw,
        }
        for label, mod in refused_mods.items():
            result, name = request(folder, LOAD_DESCRIPTOR + set_s13 + mod + DEFINE_V + plain_store + EXPORT_POSITION)
            check('%s is refused' % label, result.startswith('unsupported')
                  and fail_reason(name) == 'buffer store through a modified descriptor', '%s %s' % (result, fail_reason(name)))

        # 4. Control: a program with only a load keeps manifest version 1 and no writable view.
        load = encode_mubuf(MUBUF_LOAD_DWORD, vdata=6, srsrc=32)
        result, name = request(folder, LOAD_DESCRIPTOR + load + EXPORT_POSITION)
        check('load-only control translates', result == 'ok', '%s %s' % (result, fail_reason(name)))
        if result == 'ok':
            check('load-only stays version 1',
                  name.with_suffix('.txt').read_text().startswith('x2w-manifest 1\n')
                  and 'RWByteAddressBuffer' not in name.with_suffix('.vs.hlsl').read_text())

        # 5. Forms the runtime slice does not cover stay refused.
        refused = {
            'glc': encode_mubuf(STORE_OPS['dwordx3'], glc=1),
            'slc': encode_mubuf(STORE_OPS['dwordx3'], slc=1),
            'offen': encode_mubuf(STORE_OPS['dwordx3'], offen=1, vaddr=4),
            'idxen': encode_mubuf(STORE_OPS['dwordx3'], idxen=1, vaddr=4),
        }
        for label, code in refused.items():
            result, name = request(folder, LOAD_DESCRIPTOR + DEFINE_V + code + EXPORT_POSITION)
            check('%s store is refused' % label, result.startswith('unsupported') and not name.with_suffix('.txt').exists(),
                  '%s %s' % (result, fail_reason(name)))

        # 6. A descriptor used for both a load and a store is refused, in either order.
        for label, body in (('store then load', encode_mubuf(STORE_OPS['dwordx3']) + load),
                            ('load then store', load + encode_mubuf(STORE_OPS['dwordx3']))):
            result, name = request(folder, LOAD_DESCRIPTOR + DEFINE_V + body + EXPORT_POSITION)
            check('%s on one descriptor is refused' % label,
                  result.startswith('unsupported') and 'both read and stored to' in fail_reason(name), '%s %s' % (result, fail_reason(name)))

        # 7. A pixel-stage store is refused.
        ps = LOAD_DESCRIPTOR + DEFINE_V + encode_mubuf(STORE_OPS['dwordx3']) + PS
        result, name = request(folder, LOAD_DESCRIPTOR + EXPORT_POSITION, ps)
        check('pixel-stage store is refused', result.startswith('unsupported') and not name.with_suffix('.txt').exists(),
              '%s %s' % (result, fail_reason(name)))

        # 8. The cache revision moved, so a translation from before the store lowering is never loaded.
        check('cache key uses revision 28', G.pipeline_key(b'a', b'b', ROOT) ==
              G.fnv(ROOT, G.fnv(b'b', G.fnv(b'a', G.fnv(b'x2w-startup-shader-29')))))

    if failures:
        print('FAILED: ' + ', '.join(failures))
        return 1
    print('ok')
    return 0


sys.exit(main())
