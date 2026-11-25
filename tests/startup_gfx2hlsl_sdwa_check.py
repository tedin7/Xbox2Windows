"""Check VOP1/VOP2 SDWA translation in tools/startup/gfx2hlsl.py (#1141, #2274).

usage: startup_gfx2hlsl_sdwa_check.py

Sub-dword addressing selects a byte or word out of a source, optionally sign-extended, and writes the
low bits of the result back into a byte or word of the destination. The tools must agree on that
meaning, so this drives the translator on gfx1013 assembly and compares every helper result against
tools/pixel_aot.py, which already lowers the same forms.

Pixel AOT refuses a destination select that needs a tracked numeric destination; a register variable
in this translator is always numeric, so only the PAD and PRESERVE forms are compared there.
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402
import pixel_aot as P  # noqa: E402

SELS = ['BYTE_0', 'BYTE_1', 'BYTE_2', 'BYTE_3', 'WORD_0', 'WORD_1', 'DWORD']
failures = []


def check(label, condition):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)


# 1. The selection helpers mean the same thing as the pixel AOT ones.
mismatched = [(sel, sext) for sel in SELS for sext in (False, True)
              if G.sdwa_source('x', sel, sext) != P.sdwa_source('x', sel, sext)]
check('source selection parity', not mismatched)
if mismatched:
    print('  differs:', mismatched)
mismatched = [(sel, unused) for sel in SELS for unused in ('UNUSED_PAD', 'UNUSED_PRESERVE')
              if G.sdwa_dest('r', sel, unused, 'alu1') != P.sdwa_dest('r', sel, unused, 'alu1')]
check('destination selection parity', not mismatched)
if mismatched:
    print('  differs:', mismatched)


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for one instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.masks = set()


def translate(assembly):
    stage = Probe()
    stage.one(assembly.split(' ', 1)[0], assembly)
    return ' '.join(stage.lines)


# 2. Byte and word selects reach the emitted HLSL.
emitted = translate('v_add_nc_u32_sdwa v1, v2, v3 dst_sel:BYTE_1 dst_unused:UNUSED_PAD '
                    'src0_sel:BYTE_2 src1_sel:WORD_1')
check('byte destination select emitted', '<< 8u' in emitted and 'v1 =' in emitted)
check('byte source select emitted', '(v2 >> 16u) & 0xffu' in emitted)
check('word source select emitted', '(v3 >> 16u) & 0xffffu' in emitted)
check('operation grouped before the destination mask', '((v2 >> 16u) & 0xffu) + ' in emitted)
print('  emitted:', emitted)

# 3. sext() sign-extends the selected field instead of zero-extending it.
emitted = translate('v_and_b32_sdwa v4, sext(v5), v0 dst_sel:DWORD dst_unused:UNUSED_PAD '
                    'src0_sel:BYTE_0 src1_sel:DWORD')
check('sext emitted as a shift pair', 'asuint(asint(v5 << 24u) >> 24u)' in emitted)
print('  emitted:', emitted)

# 4. UNUSED_PRESERVE keeps the rest of the destination.
emitted = translate('v_or_b32_sdwa v6, v7, v8 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE '
                    'src0_sel:DWORD src1_sel:DWORD')
check('preserve keeps the destination', 'asuint(v6)' in emitted and '0xffffu' in emitted)
print('  emitted:', emitted)

# 5. A DWORD selection is the identity, so the operation is unchanged.
emitted = translate('v_xor_b32_sdwa v9, v10, v11 dst_sel:DWORD dst_unused:UNUSED_PAD '
                    'src0_sel:DWORD src1_sel:DWORD')
check('dword select is the identity', emitted.replace(' ', '') == 'v9=v10^v11;')
print('  emitted:', emitted)


# 6. Forms with no lowering are refused rather than mistranslated.
def refused(assembly):
    try:
        translate(assembly)
    except G.Unsupported:
        return True
    return False


check('float SDWA refused', refused('v_add_f32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD '
                                    'src0_sel:DWORD src1_sel:DWORD'))
check('UNUSED_SEXT refused', refused('v_add_nc_u32_sdwa v1, v2, v3 dst_sel:BYTE_0 '
                                     'dst_unused:UNUSED_SEXT src0_sel:DWORD src1_sel:DWORD'))
check('missing source select refused', refused('v_add_nc_u32_sdwa v1, v2, v3 dst_sel:DWORD '
                                               'dst_unused:UNUSED_PAD src0_sel:DWORD'))

# 7. v_mov_b32_sdwa (#2274): the selected source field lands in the destination field. Every expected
# statement is a literal, so a wrong shift, mask or keep-mask in any selection fails its own case.
def refused_as(assembly, reason):
    try:
        translate(assembly)
    except G.Unsupported as error:
        return str(error) == reason
    return False


def mov(source, dst_sel, unused, src_sel):
    try:
        return translate('v_mov_b32_sdwa v1, %s dst_sel:%s dst_unused:%s src0_sel:%s'
                         % (source, dst_sel, unused, src_sel))
    except G.Unsupported as error:
        return 'refused: %s' % error


MOV_ZERO = {'BYTE_0': '((v2 >> 0u) & 0xffu)', 'BYTE_1': '((v2 >> 8u) & 0xffu)', 'BYTE_2': '((v2 >> 16u) & 0xffu)',
            'BYTE_3': '((v2 >> 24u) & 0xffu)', 'WORD_0': '((v2 >> 0u) & 0xffffu)',
            'WORD_1': '((v2 >> 16u) & 0xffffu)', 'DWORD': 'v2'}
MOV_SEXT = {'BYTE_0': 'asuint(asint(v2 << 24u) >> 24u)', 'BYTE_1': 'asuint(asint(v2 << 16u) >> 24u)',
            'BYTE_2': 'asuint(asint(v2 << 8u) >> 24u)', 'BYTE_3': 'asuint(asint(v2 << 0u) >> 24u)',
            'WORD_0': 'asuint(asint(v2 << 16u) >> 16u)', 'WORD_1': 'asuint(asint(v2 << 0u) >> 16u)', 'DWORD': 'v2'}
MOV_PAD = {'BYTE_0': '(((v2) & 0xffu) << 0u)', 'BYTE_1': '(((v2) & 0xffu) << 8u)', 'BYTE_2': '(((v2) & 0xffu) << 16u)',
           'BYTE_3': '(((v2) & 0xffu) << 24u)', 'WORD_0': '(((v2) & 0xffffu) << 0u)',
           'WORD_1': '(((v2) & 0xffffu) << 16u)', 'DWORD': 'v2'}
MOV_KEEP = {'BYTE_0': '0xffffff00u', 'BYTE_1': '0xffff00ffu', 'BYTE_2': '0xff00ffffu', 'BYTE_3': '0xffffffu',
            'WORD_0': '0xffff0000u', 'WORD_1': '0xffffu'}
for sel in SELS:
    preserved = 'v2' if sel == 'DWORD' else '((asuint(v1) & %s) | %s)' % (MOV_KEEP[sel], MOV_PAD[sel])
    for label, emitted, expected in (
            ('source %s zero-extended' % sel, mov('v2', 'DWORD', 'UNUSED_PAD', sel), MOV_ZERO[sel]),
            ('source %s sign-extended' % sel, mov('sext(v2)', 'DWORD', 'UNUSED_PAD', sel), MOV_SEXT[sel]),
            ('destination %s UNUSED_PAD' % sel, mov('v2', sel, 'UNUSED_PAD', 'DWORD'), MOV_PAD[sel]),
            ('destination %s UNUSED_PRESERVE' % sel, mov('v2', sel, 'UNUSED_PRESERVE', 'DWORD'), preserved)):
        check('v_mov ' + label, emitted == 'v1 = %s;' % expected)
        if emitted != 'v1 = %s;' % expected:
            print('  emitted:', emitted)
check('v_mov source and destination selections compose',
      mov('sext(v2)', 'BYTE_1', 'UNUSED_PRESERVE', 'WORD_1') ==
      'v1 = ((asuint(v1) & 0xffff00ffu) | (((asuint(asint(v2 << 0u) >> 16u)) & 0xffu) << 8u));')
check('v_mov scalar source', mov('s3', 'DWORD', 'UNUSED_PAD', 'BYTE_1') == 'v1 = ((s3 >> 8u) & 0xffu);')
check('v_mov inline constant source',
      mov('-16', 'DWORD', 'UNUSED_PAD', 'BYTE_0') == 'v1 = ((0xfffffff0u >> 0u) & 0xffu);')
check('v_mov UNUSED_SEXT with a DWORD destination has no unused bits',
      mov('v2', 'DWORD', 'UNUSED_SEXT', 'WORD_0') == 'v1 = ((v2 >> 0u) & 0xffffu);')
check('v_mov whole lane mask is the plain move', mov('exec_lo', 'DWORD', 'UNUSED_PAD', 'DWORD') == 'v1 = exec;')
check('v_mov UNUSED_SEXT with a sub-dword destination refused',
      refused_as('v_mov_b32_sdwa v1, v2 dst_sel:WORD_0 dst_unused:UNUSED_SEXT src0_sel:DWORD', 'SDWA UNUSED_SEXT'))
check('v_mov clamp refused', refused_as('v_mov_b32_sdwa v1, v2 clamp dst_sel:DWORD dst_unused:UNUSED_PAD '
                                        'src0_sel:DWORD', 'v_mov_b32 with SDWA modifier clamp'))
check('v_mov without src0_sel refused', refused_as('v_mov_b32_sdwa v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD',
                                                   'v_mov_b32 SDWA without src0_sel'))
check('v_mov without dst_sel refused', refused_as('v_mov_b32_sdwa v1, v2 dst_unused:UNUSED_PAD src0_sel:DWORD',
                                                  'v_mov_b32 SDWA without dst_sel'))
check('v_mov byte of a lane mask refused',
      refused_as('v_mov_b32_sdwa v1, exec_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0',
                 'v_mov_b32 SDWA BYTE_0 of lane mask exec_lo'))
check('v_mov sign-extended word of a lane mask refused',
      refused_as('v_mov_b32_sdwa v1, sext(vcc_lo) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0',
                 'v_mov_b32 SDWA WORD_0 of lane mask sext(vcc_lo)'))
check('SDWA form outside the tables still refused',
      refused_as('v_rcp_f32_sdwa v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD', 'SDWA form of v_rcp_f32'))

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')