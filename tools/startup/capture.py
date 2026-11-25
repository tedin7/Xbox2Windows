"""Reader for capture.bin written by the frame capture (startup tools; see docs/startup-runtime.md)."""
import argparse
import itertools
import struct
import sys
from collections import defaultdict, Counter
from pathlib import Path

LIST_SLOTS = {
    9: 'Close', 12: 'DrawInstanced', 13: 'DrawIndexedInstanced', 14: 'Dispatch', 20: 'IASetPrimitiveTopology',
    21: 'RSSetViewports', 22: 'RSSetScissorRects', 23: 'OMSetBlendFactor', 24: 'OMSetStencilRef', 25: 'SetPipelineState',
    32: 'SetDescriptorHeaps', 33: 'SetComputeRootSignature', 34: 'SetGraphicsRootSignature',
    35: 'SetComputeRootDescriptorTable', 36: 'SetGraphicsRootDescriptorTable', 39: 'SetComputeRoot32BitConstants',
    40: 'SetGraphicsRoot32BitConstants', 41: 'SetComputeRootCBV', 42: 'SetGraphicsRootCBV', 43: 'SetComputeRootSRV',
    44: 'SetGraphicsRootSRV', 45: 'SetComputeRootUAV', 46: 'SetGraphicsRootUAV', 47: 'IASetIndexBuffer',
    48: 'IASetVertexBuffers', 50: 'OMSetRenderTargets', 51: 'ClearDepthStencilView', 52: 'ClearRenderTargetView',
    139: 'Reset', 146: 'Barrier?',
}
KINDS = None  # filled lazily by callers that care


class Memory:
    """Sparse title memory: every MEM chunk, addressable by title address and capture sequence."""

    def __init__(self):
        self.blocks = {}  # address -> [(seq, bytes)] in capture order

    def add(self, address, data, seq=0):
        self.blocks.setdefault(address, []).append((seq, data))

    def read(self, address, size, seq=None):
        """Bytes at address as last captured at or before call `seq` (None: latest); None if unknown."""
        best, best_seq = None, -1
        for base, versions in self.blocks.items():
            if not (base <= address < base + len(versions[0][1]) or any(base <= address < base + len(d) for _, d in versions)):
                continue
            for s, data in versions:
                if seq is not None and s > seq:
                    continue
                if base <= address < base + len(data) and s >= best_seq:
                    piece = data[address - base:address - base + size]
                    if s > best_seq or best is None or len(piece) > len(best):
                        best, best_seq = piece, s
        return best


class Capture:
    def __init__(self, path):
        self.calls = []       # dict(frame, seq, tid, slot, list, args[7], ret)
        self.memory = Memory()
        self.mem_by_seq = defaultdict(list)
        self.psos = {}        # wrapper -> dict(base, path, real, desc)
        self.roots = {}       # wrapper -> blob
        self.resources = {}   # wrapper -> dict
        self.heaps = {}       # wrapper -> dict
        self.objects = {}     # wrapper -> kind
        self.views = {}       # id -> dict
        self.tables = defaultdict(list)   # seq -> [dict]
        self.targets = {}     # seq -> [view id + 1]
        self.presents = []
        with open(path, 'rb') as stream:
            data = stream.read()
        at = 0
        while at < len(data):
            start = at
            if at + 16 > len(data):
                raise ValueError('truncated chunk header at offset %d' % start)
            tag, hsize, dsize = struct.unpack_from('<4sIQ', data, at)
            at += 16
            if hsize + dsize > len(data) - at:
                raise ValueError('truncated %r chunk at offset %d: declares %d header and %d data bytes, %d left' % (
                    tag, start, hsize, dsize, len(data) - at))
            header = data[at:at + hsize]
            at += hsize
            body = data[at:at + dsize]
            at += dsize
            handler = getattr(self, 'on_' + tag.decode('ascii', 'replace').strip(), None)
            if handler is None:
                raise ValueError('unknown chunk tag %r at offset %d' % (tag, start))
            try:
                handler(header, body)
            except struct.error as error:
                raise ValueError('bad %r chunk at offset %d: %s' % (tag, start, error)) from error

    def on_FRAM(self, h, b):
        if len(h) < struct.calcsize('<II'):
            raise struct.error('frame header requires the frame number and reserved word')

    def on_CALL(self, h, b):
        frame, seq, tid, slot, lst = struct.unpack_from('<IIIIQ', h, 0)
        args = struct.unpack_from('<7Q', h, 24)
        ret, = struct.unpack_from('<Q', h, 80)
        self.calls.append(dict(frame=frame, seq=seq, tid=tid, slot=slot, list=lst, args=args, ret=ret))

    def on_MEM(self, h, b):
        address, size, wanted, why, seq = struct.unpack_from('<QQQII', h, 0)
        self.memory.add(address, b, seq)
        self.mem_by_seq[seq].append((address, len(b), why))

    def on_PSO(self, h, b):
        wrapper, base, path, real = struct.unpack_from('<QQII', h, 0)
        self.psos[wrapper] = dict(base=base, path=path, real=real, desc=b)
        self.objects[wrapper] = 'PipelineState'

    def on_ROOT(self, h, b):
        wrapper, = struct.unpack_from('<Q', h, 0)
        self.roots[wrapper] = b
        self.objects[wrapper] = 'RootSignature'

    def on_RES(self, h, b):
        wrapper, title_memory, size, gpu = struct.unpack_from('<4Q', h, 0)

        def desc(at):
            # D3D12_RESOURCE_DESC: Dimension u32, pad, Alignment u64, Width u64, Height u32, Depth u16, Mips u16,
            # Format u32, SampleDesc {u32, u32}, Layout u32, Flags u32  = 56 bytes
            dim, = struct.unpack_from('<I', h, at)
            align, width = struct.unpack_from('<QQ', h, at + 8)
            height, depth, mips, fmt, samples, quality, layout, flags = struct.unpack_from('<IHHIIIII', h, at + 24)
            return dict(dim=dim, align=align, width=width, height=height, depth=depth, mips=mips, format=fmt,
                        samples=samples, layout=layout, flags=flags)
        self.resources[wrapper] = dict(title_memory=title_memory, bytes=size, gpu=gpu, console=desc(32), desktop=desc(88))
        self.objects[wrapper] = 'Resource'

    def on_HEAP(self, h, b):
        wrapper, typ, count, inc, pad, title_cpu, desktop_cpu, desktop_gpu = struct.unpack_from('<QIIIIQQQ', h, 0)
        self.heaps[wrapper] = dict(type=typ, count=count, increment=inc, title_cpu=title_cpu, desktop_cpu=desktop_cpu,
                                   desktop_gpu=desktop_gpu)
        self.objects[wrapper] = 'DescriptorHeap'

    def on_OBJ(self, h, b):
        wrapper, real, title_memory, title_cpu, kind = struct.unpack_from('<QQQQI', h, 0)
        self.objects[wrapper] = 'kind%d' % kind

    def on_VIEW(self, h, b):
        vid, slot, resource, counter, dest_title, dest_desktop, desc_bytes = struct.unpack_from('<IIQQQQI', h, 0)
        if len(h) < 48 + desc_bytes:
            raise ValueError('truncated VIEW descriptor: header %d bytes, expected at least %d' % (
                len(h), 48 + desc_bytes))
        self.views[vid] = dict(slot=slot, resource=resource, counter=counter, dest_title=dest_title,
                               desc=h[48:48 + desc_bytes])

    def on_TABL(self, h, b):
        seq, slot, parameter, entries, gpu, heap, index = struct.unpack_from('<IIIIQQQ', h, 0)
        if len(b) % 4 != 0:
            raise ValueError('malformed TABL body: %d bytes is not a multiple of 4' % len(b))
        ids = struct.unpack_from('<%dI' % (len(b) // 4), b, 0)
        if entries > len(ids):
            raise ValueError('truncated TABL body: declares %d entries, got %d ids' % (entries, len(ids)))
        self.tables[seq].append(dict(slot=slot, parameter=parameter, entries=entries, gpu=gpu, heap=heap, index=index,
                                     shadow=ids[:entries], tokens=ids[entries:]))

    def on_TGTS(self, h, b):
        seq, count = struct.unpack_from('<II', h, 0)
        self.targets[seq] = struct.unpack_from('<%dI' % count, b, 0)

    def on_PRES(self, h, b):
        frame, pad, surface = struct.unpack_from('<IIQ', h, 0)
        self.presents.append((frame, surface))

    def name(self, value):
        kind = self.objects.get(value)
        return '%s@%x' % (kind, value & 0xffffff) if kind else None


def describe(cap, out=sys.stdout):
    w = out.write
    w('calls %d, memory blocks %d (%d bytes), pipelines %d, roots %d, resources %d, heaps %d, views %d\n' % (
        len(cap.calls), len(cap.memory.blocks), sum(len(d) for v in cap.memory.blocks.values() for _, d in v), len(cap.psos),
        len(cap.roots), len(cap.resources), len(cap.heaps), len(cap.views)))
    w('slot histogram: %s\n' % ', '.join('%s x%d' % (LIST_SLOTS.get(s, 'slot%d' % s), n)
                                          for s, n in Counter(c['slot'] for c in cap.calls).most_common()))
    for c in cap.calls:
        args = []
        for i, a in enumerate(c['args'][:3]):
            args.append(cap.name(a) or ('0x%x' % a))
        w('%4d f%d L%03x %-30s %s | %s\n' % (c['seq'], c['frame'], c['list'] & 0xfff, LIST_SLOTS.get(c['slot'], 'slot%d' % c['slot']),
                                             ', '.join(args), ' '.join('0x%x' % a for a in c['args'][3:5])))
        for t in cap.tables.get(c['seq'], ()):
            used = [(i, s, k) for i, (s, k) in enumerate(itertools.zip_longest(t['shadow'], t['tokens'], fillvalue=0)) if s or k]
            w('       table param %d heap index %d: %d of %d entries known\n' % (t['parameter'] if t['parameter'] != 0xffffffff else -1,
                                                                             t['index'], len(used), t['entries']))
            for i, s, k in used[:64]:
                v = cap.views.get((s or k) - 1)
                if not v:
                    w('         [%2d] view %d (not recorded)\n' % (i, (s or k) - 1))
                    continue
                r = cap.resources.get(v['resource'])
                rd = ''
                if r:
                    d = r['console']
                    rd = ' dim%d %dx%dx%d mips%d fmt%d layout0x%x flags0x%x mem=0x%x' % (
                        d['dim'], d['width'], d['height'], d['depth'], d['mips'], d['format'], d['layout'], d['flags'], r['title_memory'])
                w('         [%2d] %s view %d%s res=%x%s desc=%s\n' % (
                    i, {17: 'CBV', 18: 'SRV', 19: 'UAV', 20: 'RTV', 21: 'DSV', 22: 'SMP'}.get(v['slot'], '?'), (s or k) - 1,
                    '' if s == k or not t['tokens'] else ' (shadow %d token %d)' % (s - 1, k - 1), v['resource'] & 0xffffff, rd, v['desc'][:24].hex()))
        if c['seq'] in cap.targets:
            for s in cap.targets[c['seq']]:
                v = cap.views.get(s - 1)
                r = cap.resources.get(v['resource']) if v else None
                if r:
                    d = r['console']
                    w('       target view %d res=%x %dx%d fmt%d flags0x%x\n' % (s - 1, v['resource'] & 0xffffff, d['width'], d['height'], d['format'], d['flags']))
                else:
                    w('       target view %d (unknown)\n' % (s - 1))
    for frame, surface in cap.presents:
        w('present frame %d surface %s\n' % (frame, cap.name(surface)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path, help='Path to capture.bin')
    args = parser.parse_args(argv)
    try:
        describe(Capture(args.capture), out=sys.stdout)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
