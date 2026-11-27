#!/usr/bin/env python3
"""Compute LDS b64 ordered-pair controls and D3D12 readback."""
import hashlib
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_compute_lds_b64_check as sibling
from startup_gfx2hlsl_compute_check import patch, refused, renderer_diagnostics

WORDS = sibling.WORDS
SENTINEL = sibling.SENTINEL
BASE = patch(sibling.WORDS_P, '00008abf', '000080bf')
OOB = patch(BASE, '1000d8d908000008', '0004d8d908000008')
SAME_WAVE = patch(BASE, 'ff08103a00010000', 'ff08103a08000000')


def reference(out_of_bounds=False, wrong_peer=False, same_wave=False):
    # AMD RDNA 2 ISA guide, 10.4.3 and 12.13: byte-addressed, little-endian LDS.
    result = [SENTINEL] * WORDS
    for group in range(2):
        lds = bytearray(1024)
        for wave in range(2):
            for lane in range(wave * 32, wave * 32 + 32):
                value = (group << 8) + lane
                struct.pack_into('<II', lds, lane * 8 + 16, value, value ^ 0xa5a50000)
        for lane in range(64):
            peer = lane if wrong_peer else lane ^ (1 if same_wave else 32)
            address = 1024 if out_of_bounds else peer * 8 + 16
            for word in range(2):
                offset = address + word * 4
                value = struct.unpack_from('<I', lds, offset)[0] if offset + 4 <= len(lds) else 0
                result[(group * 64 + lane) * 2 + word] = value
    return result


def shader(code=BASE):
    return sibling.translate_case('P', code=code).hlsl_cs()


# Hand-encoded source-built programs. Every address in the multi-wave cases is
# either wave-private or separated by a group barrier.
import startup_compute_lds_dual_atomics_check as enc
import gfx2hlsl as G


def program(name):
    p = [raw for _, raw in enc.PREAMBLE_LINES]
    def v(op, dst, src, other, lit=None):
        p.append(enc.vop2(op, dst, src, other, lit))
    def ds(op, addr, data=0, dst=0, off=0):
        p.append(enc.ds_single(op, addr, data0=data, vdst=dst, offset=off))
    def dual(op, addr, d0=0, d1=0, dst=0, off1=1):
        p.append(enc.ds_dual(op, addr, data0=d0, data1=d1, vdst=dst, offset1=off1))
    group = (8, 8, 1) if name == 'transpose' else (64, 1, 1)
    if name == 'transpose':
        v(0x1a, 3, 0x83, 1)             # y * 8
        v(0x25, 3, 0x100, 3)            # flat = x + 8*y
    else:
        p.append(enc.vop1(1, 3, 0x100))  # flat = x
    v(0x1a, 4, 0x82, 3)                 # 4*flat
    if name in ('transpose', 'same_wave', 'bounds'):
        if name == 'bounds':
            v(0x25, 5, 0x81, 3)
            ds(0x0d, 4, data=5)
            ds(0x0d, 4, data=3, off=2048)  # dropped
            ds(0x36, 4, dst=20, off=2048)  # zero
            ds(0x36, 4, dst=21)
        else:
            ds(0x0d, 4, data=3)
            if name == 'transpose':
                p.append(enc.sopp(10))
                v(0x1a, 5, 0x83, 0)
                v(0x25, 5, 0x101, 5)    # y + 8*x
            else:
                v(0x1d, 5, 0x81, 3)     # flat ^ 1
            v(0x1a, 5, 0x82, 5)
            ds(0x36, 5, dst=20)
        v(0x1a, 13, 0x83 if name == 'bounds' else 0x82, 3)
        p.append(enc.mubuf_store(0x1d if name == 'bounds' else 0x1c, 20, 13))
    elif name == 'unaligned':
        v(0x1a, 6, 0x83, 3)
        v(0x25, 6, 0x81, 6)
        ds(0x0d, 6, data=3)
        ds(0x36, 6, dst=20)
        v(0x1a, 13, 0x82, 3)
        p.append(enc.mubuf_store(0x1c, 20, 13))
    elif name == 'bytes':
        v(0x25, 5, 0xff, 3, 128)
        ds(0x1e, 3, data=5)               # each lane writes one byte
        v(0x1a, 6, 0x81, 3)
        v(0x25, 6, 0xff, 6, 256)
        ds(0x1f, 6, data=3)               # adjacent halfwords
        v(0x1b, 7, 0xff, 3, 0xfffffffc)
        ds(0x36, 7, dst=20)
        p.append(enc.v_mov_imm(21, 0x12345678))
        ds(0xa7, 6, dst=21)
        ds(0x39, 3, dst=22)
        p.append(enc.v_mov_imm(23, 0))
        v(0x1a, 13, 0x84, 3)
        p.append(enc.mubuf_store(0x1e, 20, 13))
    elif name == 'dual_wide':
        v(0x1d, 5, 0xff, 3, 0x10000)
        dual(0x0f, 4, d0=3, d1=5)
        p.append(enc.sopp(10))
        dual(0x37, 4, dst=20)
        v(0x1a, 8, 0x84, 3)
        v(0x25, 8, 0xff, 8, 1024)
        p.append(enc.vop1(1, 9, 0x103))
        p.append(enc.vop1(1, 10, 0x105))
        p.append(enc.vop1(1, 11, 0x103))
        p.append(enc.vop1(1, 12, 0x105))
        ds(0xdf, 8, data=9)
        ds(0x76, 8, dst=22)
        v(0x1a, 13, 0x84, 3)
        p.append(enc.mubuf_store(0x1e, 20, 13))
    else:
        raise ValueError(name)
    p.append(enc.sopp(1))
    rsrc = sibling.USER7 | sibling.TGID_X | (4 << 15) | ((1 << 11) if name == 'transpose' else 0)
    return b''.join(p), rsrc, group


def simulate(name, reverse=False):
    """Run each instruction for a wave's lanes in program order between barriers."""
    mem = bytearray(2048)
    reg = [[0] * 4 for _ in range(64)]
    def put(addr, value, size):
        for i in range(size):
            if addr + i < len(mem):
                mem[addr + i] = (value >> (8 * i)) & 255
    def get(addr, size):
        return sum((mem[addr + i] if addr + i < len(mem) else 0) << (8 * i) for i in range(size))
    def run(stages):
        for stage in stages:
            for wave in ([1, 0] if reverse else [0, 1]):
                for instruction in stage:
                    for lane in range(wave * 32, wave * 32 + 32):
                        instruction(lane)
    if name == 'transpose':
        run([[lambda l: put(l * 4, l, 4)],
             [lambda l: reg[l].__setitem__(0, get(((l % 8) * 8 + l // 8) * 4, 4))]])
        count = 1
    elif name == 'same_wave':
        run([[lambda l: put(l * 4, l, 4), lambda l: reg[l].__setitem__(0, get((l ^ 1) * 4, 4))]])
        count = 1
    elif name == 'bounds':
        run([[lambda l: put(l * 4, l + 1, 4)], [lambda l: put(2048 + l * 4, l, 4)],
             [lambda l: reg[l].__setitem__(0, get(2048 + l * 4, 4)),
              lambda l: reg[l].__setitem__(1, get(l * 4, 4))]])
        count = 2
    elif name == 'unaligned':
        run([[lambda l: put(8 * l + 1, l, 4)],
             [lambda l: reg[l].__setitem__(0, get(8 * l + 1, 4))]])
        count = 1
    elif name == 'bytes':
        run([[lambda l: put(l, l + 128, 1)], [lambda l: put(256 + 2 * l, l, 2)],
             [lambda l: reg[l].__setitem__(0, get(l & ~3, 4)),
              lambda l: reg[l].__setitem__(1, 0x5678 | (get(256 + 2 * l, 2) << 16)),
              lambda l: reg[l].__setitem__(2, (get(l, 1) | 0xffffff00))]])
        count = 4
    elif name == 'dual_wide':
        run([[lambda l: (put(4*l, l, 4), put(256+4*l, l ^ 0x10000, 4))],
             [lambda l: reg[l].__setitem__(0, get(4*l, 4)),
              lambda l: reg[l].__setitem__(1, get(4*l+4, 4))],
             [lambda l: (put(1024+16*l, l, 4), put(1028+16*l, l ^ 0x10000, 4),
                         put(1032+16*l, l, 4), put(1036+16*l, l ^ 0x10000, 4))],
             [lambda l: reg[l].__setitem__(2, get(1024+16*l, 4)),
              lambda l: reg[l].__setitem__(3, get(1028+16*l, 4))]])
        count = 4
    else:
        raise ValueError(name)
    result = [SENTINEL] * WORDS
    for lane in range(64):
        result[lane * count:lane * count + count] = reg[lane][:count]
    return result


def pipeline(name):
    code, rsrc, group = program(name)
    return G.ComputePipeline(code, sibling.root_blob(), rsrc, group)

BRANCH_SENTINEL = 0xdecafbad


def branch_program(name, mutation=None):
    p = [raw for _, raw in enc.PREAMBLE_LINES]
    p += [enc.vop1(1, 3, 0x100), enc.vop2(0x1a, 4, 0x82, 3),
          enc.ds_single(0x0d, 4, data0=3), enc.sopp(10)]
    if name == 'execz':
        p += [enc.v_mov_imm(20, BRANCH_SENTINEL), enc.vopc(0xc4, 0xa0, 0),
              bytes.fromhex('6a2480be')]
        inner = []
        if mutation == 'guest-barrier':
            inner.append(enc.sopp(10))
        if mutation == 'ordering':
            inner.append(enc.ds_single(0x0d, 4, data0=3))
        inner.append(enc.ds_single(0x36, 4, vdst=20))
        p += [enc.sopp(8, sum(len(item) for item in inner) // 4), *inner,
              bytes.fromhex('7e00fe88')]
        # A second phase after the restore tests convergence of the join.
        p += [enc.ds_single(0x0d, 4, data0=3), enc.sopp(10),
              enc.ds_single(0x36, 4, vdst=21)]
        p += [enc.vop2(0x1a, 13, 0x83, 3), enc.mubuf_store(0x1d, 20, 13)]
    elif name == 'uniform':
        p += [enc.vop2(0x25, 5, 0xff, 3, 123), bytes.fromhex('048106bf'),
              enc.sopp(5, 2), enc.ds_single(0x0d, 4, data0=5), enc.sopp(10),
              enc.ds_single(0x36, 4, vdst=20), enc.vop2(0x1a, 13, 0x82, 3),
              enc.mubuf_store(0x1c, 20, 13)]
    else:
        raise ValueError(name)
    p.append(enc.sopp(1))
    if mutation == 'loop':
        p.insert(len(enc.PREAMBLE_LINES), enc.sopp(2, 0xffff))
    rsrc = sibling.USER7 | sibling.TGID_X | (1 << 15)
    return b''.join(p), rsrc, (64, 1, 1)


def branch_pipeline(name, mutation=None):
    code, rsrc, group = branch_program(name, mutation)
    return G.ComputePipeline(code, sibling.root_blob(), rsrc, group)


def branch_reference(name, root_constant=0, reverse=False):
    """Simulate each wave in instruction order, synchronizing at guest and planned barriers."""
    result = [SENTINEL] * WORDS
    lds = bytearray(512)
    waves = (1, 0) if reverse else (0, 1)
    lanes = lambda wave: range(wave * 32, wave * 32 + 32)
    regs = [BRANCH_SENTINEL] * 64
    for wave in waves:
        for lane in lanes(wave):
            struct.pack_into('<I', lds, lane * 4, lane)
    if name == 'execz':
        for wave in waves:
            for lane in lanes(wave):
                if lane < 32:
                    regs[lane], = struct.unpack_from('<I', lds, lane * 4)
        # Inserted barrier before the second phase; then the guest barrier.
        for wave in waves:
            for lane in lanes(wave):
                struct.pack_into('<I', lds, lane * 4, lane)
        for wave in waves:
            for lane in lanes(wave):
                second, = struct.unpack_from('<I', lds, lane * 4)
                result[2 * lane:2 * lane + 2] = [regs[lane], second]
    elif name == 'uniform':
        if root_constant != 1:
            for wave in waves:
                for lane in lanes(wave):
                    struct.pack_into('<I', lds, lane * 4, lane + 123)
        for wave in waves:
            for lane in lanes(wave):
                result[lane], = struct.unpack_from('<I', lds, lane * 4)
    else:
        raise ValueError(name)
    return result


def controls():
    legacy = sibling.translate_case('P').hlsl_cs()
    assert hashlib.sha256(legacy.encode()).hexdigest() == '6f1c8c0cae027dc199d84a4d74774a68c243f205f740fac098524e15109eac70'
    text = shader()
    assert text.count('GroupMemoryBarrierWithGroupSync();') == 1
    assert text.index('lds[(a + 16u) >> 2] =') < text.index('GroupMemoryBarrierWithGroupSync();') < text.index('w = lds[(a + 16u) >> 2]')
    assert 'if (a < 1004u)' in text
    assert reference() != reference(wrong_peer=True)
    assert reference(same_wave=True) != reference(wrong_peer=True, same_wave=True)
    assert shader(SAME_WAVE).count('GroupMemoryBarrierWithGroupSync();') == 1
    assert reference(out_of_bounds=True)[:256] == [0] * 256
    assert shader(OOB).count('GroupMemoryBarrierWithGroupSync();') == 1
    unaligned = patch(BASE, '1000d8d908000008', '1200d8d908000008')
    assert 'Interlocked' not in shader(unaligned)  # the unaligned operation is a read
    assert '& 0xffu' in shader(unaligned)
    hidden_field = patch(BASE, '100034d904050000', '100034d904050100')
    refused('reserved or unused', lambda: shader(hidden_field))
    branched = patch(BASE, '100034d904050000', 'ffff82bf100034d904050000')
    refused('LDS general lowering: backward branch', lambda: shader(branched))
    for name in ('transpose', 'same_wave', 'bytes', 'dual_wide', 'bounds', 'unaligned'):
        text = pipeline(name).hlsl_cs()
        assert simulate(name) == simulate(name, reverse=True), name
        assert 'groupshared uint lds[512];' in text, name
    assert 'InterlockedAnd' in pipeline('bytes').hlsl_cs()
    code, rsrc, _ = program('transpose')
    cube = G.ComputePipeline(code, sibling.root_blob(), (rsrc & ~(3 << 11)) | (2 << 11), (4, 4, 4)).hlsl_cs()
    assert '[numthreads(4, 4, 4)]' in cube
    assert '[numthreads(8, 8, 1)]' in pipeline('transpose').hlsl_cs()
    assert simulate('same_wave')[:64] != list(range(64))
    assert simulate('bytes')[0] == 0x83828180
    assert simulate('bytes')[0] != 0x83000000  # lost byte updates
    assert simulate('bounds')[:2] == [0, 1]
    branch_text = branch_pipeline('execz').hlsl_cs()
    assert branch_text.count('GroupMemoryBarrierWithGroupSync();') == 3
    assert 'if (exec & 1)' in branch_text
    # Every barrier must be outside the per-thread pc dispatch, including the join.
    stack = []
    for line in branch_text[branch_text.index('    uint pc = 0;'):].splitlines():
        if 'GroupMemoryBarrierWithGroupSync();' in line:
            assert not any(stack), line
        pc_block = 'if (pc ==' in line
        for char in line:
            if char == '{':
                stack.append(pc_block)
                pc_block = False
            elif char == '}' and stack:
                stack.pop()
    assert branch_reference('execz')[0:2] == [0, 0]
    assert branch_reference('execz')[64:66] == [BRANCH_SENTINEL, 32]
    branch_pipeline('uniform').hlsl_cs()
    uniform_code, uniform_rsrc, uniform_group = branch_program('uniform')
    write = enc.ds_single(0x0d, 4, data0=5)
    read = enc.ds_single(0x36, 4, vdst=20)
    join_mutant = uniform_code.replace(write + enc.sopp(10) + read, write + read)
    assert join_mutant != uniform_code
    join_text = G.ComputePipeline(join_mutant, sibling.root_blob(), uniform_rsrc, uniform_group).hlsl_cs()
    assert join_text.count('GroupMemoryBarrierWithGroupSync();') == 2
    first_write = enc.ds_single(0x0d, 4, data0=3)
    dual_code = uniform_code.replace(first_write, enc.ds_dual(0x0e, 4, data0=3, data1=5, offset1=1))
    dual_code = dual_code.replace(write, enc.ds_dual(0x0e, 4, data0=5, data1=3, offset1=1))
    dual_code = dual_code.replace(read, enc.ds_dual(0x38, 4, vdst=20, offset1=1))
    dual_text = G.ComputePipeline(dual_code, sibling.root_blob(), uniform_rsrc, uniform_group).hlsl_cs()
    assert dual_text.count('GroupMemoryBarrierWithGroupSync();') == 2
    nonuniform, nr, ng = branch_program('execz')
    nonuniform = nonuniform.replace(enc.sopp(8, 2), enc.sopp(5, 2))
    refused('LDS scalar branch needs proven uniform SCC', lambda: G.ComputePipeline(nonuniform, sibling.root_blob(), nr, ng))
    missing_restore = nonuniform.replace(enc.sopp(5, 2), enc.sopp(8, 2)).replace(
        bytes.fromhex('7e00fe88'), enc.sopp(0))
    refused('LDS ordering point inside per-lane control flow', lambda: G.ComputePipeline(missing_restore, sibling.root_blob(), nr, ng))
    assert branch_reference('uniform', 0)[0] == 123
    assert branch_reference('uniform', 1)[0] == 0
    for name, value in (('execz', 0), ('uniform', 0), ('uniform', 1)):
        assert branch_reference(name, value) == branch_reference(name, value, reverse=True)
    refused('LDS barrier inside per-lane control flow', lambda: branch_pipeline('execz', 'guest-barrier'))
    refused('LDS ordering point inside per-lane control flow', lambda: branch_pipeline('execz', 'ordering'))
    refused('LDS general lowering: backward branch', lambda: branch_pipeline('execz', 'loop'))
    # The plan must split all plain/atomic hazards, but not read/read or atomic/atomic.
    assembly = ['ds_write_b32 v4, v3', 'ds_read_b32 v20, v4',
                'ds_read_b32 v21, v4', 'ds_add_u32 v4, v3',
                'ds_or_b32 v4, v3', 'ds_write_b32 v4, v3',
                's_barrier', 'ds_read_b32 v20, v4', 's_endpgm']
    ins = [(i * 4, 4, line.split()[0], line) for i, line in enumerate(assembly)]
    assert G.lds_general_plan(ins, (8, 8, 1), 512) == {1, 3, 5}
    for bad in ('s_cbranch_execz label_0004', 'ds_swizzle_b32 v1, v0', 'ds_write_b32 v4, v3 gds'):
        bad_ins = ins[:1] + [(4, 4, bad.split()[0], bad)] + ins[1:]
        assert G.lds_general_plan(bad_ins, (8, 8, 1), 512) is None
    assert G.lds_general_plan(ins, (4, 4, 4), 8193) is None
    # Every declined planning path records the actual condition and location/value.
    def plan_reason(lines, expected, group=(8, 8, 1), words=128, sites=None):
        instructions = [(i * 4, 4, line.split()[0], line) for i, line in enumerate(lines)]
        reasons = []
        assert G.lds_general_plan(instructions, group, words, sites, reasons) is None
        assert len(reasons) == 1 and expected in reasons[0], (expected, reasons)
    plan_reason(['s_endpgm'], 'allocation is 8193 dwords', words=8193)
    plan_reason(['s_endpgm'], 'invalid group shape (0, 8, 1)', group=(0, 8, 1))
    plan_reason(['s_setpc_b64 s[0:1]'], 's_setpc_b64 at +0x0')
    plan_reason(['s_cbranch_vccz label_0x4', 's_endpgm'], 'branch s_cbranch_vccz at +0x0')
    plan_reason(['s_branch label_0x40', 's_endpgm'], 'invalid branch target')
    plan_reason(['s_branch label_0x0', 's_endpgm'], 'backward branch at +0x0 to +0x0')
    plan_reason(['s_endpgm', 's_endpgm'], 'early s_endpgm at +0x0')
    plan_reason(['s_and_saveexec_b64 s[0:1], vcc', 's_endpgm'], 'EXEC write s_and_saveexec_b64 at +0x0')
    plan_reason(['s_xor_b64 exec, exec, s[0:1]', 's_endpgm'], 'EXEC write s_xor_b64 at +0x0', sites={0: 0, 4: 0})
    plan_reason(['ds_write_b32 v4, v3 gds', 's_endpgm'], 'GDS operation ds_write_b32 at +0x0')
    plan_reason(['ds_swizzle_b32 v1, v0', 's_endpgm'], 'ds_swizzle_b32 at +0x0 is not supported')
    plan_reason(['s_branch label_0x4', 's_endpgm'], 'no convergence proof for 2 instructions')
    plan_reason(['s_barrier 1', 's_endpgm'], 's_barrier at +0x0 has operands')
    plan_reason(['s_branch label_0x4', 's_barrier', 's_endpgm'], 'barrier inside per-lane control flow at +0x4', sites={8: 8})
    plan_reason(['ds_write_b32 v6, v7', 'ds_read_b32 v8, v6', 's_endpgm'],
                'ordering point inside per-lane control flow at +0x4', sites={0: 0, 8: 8})
    plan_reason(['s_branch label_0x4', 's_endpgm'], 'lacks proven group convergence and entry EXEC at +0x4', sites={})
    invalid, rsrc, group = program('same_wave')
    write = enc.ds_single(0x0d, 4, data0=3)
    refused('LDS general lowering: GDS operation', lambda: G.ComputePipeline(
        invalid.replace(write, enc.ds_single(0x0d, 4, data0=3, gds=1)), sibling.root_blob(), rsrc, group))
    refused('LDS general lowering: allocation is 8320 dwords', lambda: G.ComputePipeline(
        invalid, sibling.root_blob(), (rsrc & ~(0x1ff << 15)) | (65 << 15), group))
    unknown = invalid.replace(write, enc.ds_single(0x35, 4, vdst=20))
    refused('ds_swizzle_b32', lambda: G.ComputePipeline(unknown, sibling.root_blob(), rsrc, group))
    atomic = invalid.replace(write,
                             write + enc.ds_single(0x00, 3, data0=3))
    refused('proven 4-byte aligned address', lambda: G.ComputePipeline(atomic, sibling.root_blob(), rsrc, group))
    # Decode and translate every general opcode using the real compute entry point.
    from disasm_rdna2 import DS_OPS
    by_name = {entry[0]: op for op, entry in DS_OPS.items()}
    original, rsrc, group = program('same_wave')
    old_write = enc.ds_single(0x0d, 4, data0=3)
    old_read = enc.ds_single(0x36, 5, vdst=20)
    for name, (is_write, width, kind) in G.LDS_GENERAL_OPS.items():
        if kind == 'dual':
            raw = enc.ds_dual(by_name[name], 4 if is_write else 5,
                              data0=9 if is_write else 0, data1=13 if is_write else 0,
                              vdst=0 if is_write else 20, offset1=1)
        else:
            raw = enc.ds_single(by_name[name], 4 if is_write else 5,
                                data0=9 if is_write else 0, vdst=0 if is_write else 20)
        old = old_write if is_write else old_read
        assert original.count(old) == 1, name
        translated = G.ComputePipeline(original.replace(old, raw), sibling.root_blob(), rsrc, group).hlsl_cs()
        assert 'lds[' in translated, name
    print('general LDS translation and wave-order controls passed')


def dispatch(dxc, runner, adapter):
    root = Path(__file__).resolve().parent.parent
    out = root / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        for name, root_constant in (('execz', 0), ('uniform', 0), ('uniform', 1)):
            case_name = '%s_%d' % (name, root_constant)
            src, cso, case = (directory / (case_name + ext) for ext in ('.hlsl', '.cso', '.case'))
            src.write_text(branch_pipeline(name).hlsl_cs(), encoding='ascii')
            subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
            constants = [root_constant] + [0] * 15
            case.write_bytes(struct.pack('<I', 1) + struct.pack('<%dI' % WORDS, *branch_reference(name, root_constant)) + struct.pack('<16I', *constants))
            run = subprocess.run([str(Path(runner).resolve()), adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            diagnostics = renderer_diagnostics(run.stderr, adapter)
            if run.returncode or diagnostics:
                raise AssertionError('%s: exit %d %s' % (case_name, run.returncode, run.stderr))
        for name in ('transpose', 'same_wave', 'bytes', 'dual_wide', 'bounds', 'unaligned'):
            src, cso, case = (directory / (name + ext) for ext in ('.hlsl', '.cso', '.case'))
            src.write_text(pipeline(name).hlsl_cs(), encoding='ascii')
            subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
            case.write_bytes(struct.pack('<I', 1) + struct.pack('<%dI' % WORDS, *simulate(name)) + struct.pack('<16I', *([0] * 16)))
            run = subprocess.run([str(Path(runner).resolve()), adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            diagnostics = renderer_diagnostics(run.stderr, adapter)
            if run.returncode or diagnostics:
                raise AssertionError('%s: exit %d %s' % (name, run.returncode, run.stderr))
        for name, code, oob, same_wave in (('exchange', BASE, False, False),
                                            ('same_wave', SAME_WAVE, False, True),
                                            ('out_of_bounds', OOB, True, False)):
            src, cso, case = (directory / (name + ext) for ext in ('.hlsl', '.cso', '.case'))
            src.write_text(shader(code), encoding='ascii')
            subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
            case.write_bytes(struct.pack('<I', 2) + struct.pack('<%dI' % WORDS, *reference(oob, same_wave=same_wave)) + struct.pack('<16I', *([0] * 16)))
            run = subprocess.run([str(Path(runner).resolve()), adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            diagnostics = renderer_diagnostics(run.stderr, adapter)
            if run.returncode or diagnostics:
                raise AssertionError('%s: exit %d %s' % (name, run.returncode, run.stderr))
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        sys.exit(dispatch(*sys.argv[2:]))
    else:
        raise SystemExit('usage: startup_compute_lds_general_check.py [dispatch DXC RUNNER ADAPTER]')
