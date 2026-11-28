"""Build a two-table compute request with source-generated instructions and an SDK root signature."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def write_request(work, words, root, sa=None):
    code = struct.pack(f"<{len(words)}I", *words)
    data = b"synthetic"
    if sa is None:
        sa = struct.pack("<7I", 0, 0, 0, 32 | (8 << 16), 1 | (1 << 20), 0, (6 << 1) | (1 << 10))
    desc = struct.pack("<12Q", 0, 0, len(sa), 0, 0, 0, len(code), 0, len(data), 0, 0, 0)
    parts = {".c67":code, ".data":data, ".root":root, ".desc":desc, ".sa":sa, ".sb":b""}
    payload = b"x2w-startup-shader-28cs" + b"".join(struct.pack("<Q", len(p))+p for p in parts.values())
    key = 1469598103934665603
    for byte in payload:
        key = (key ^ byte)*1099511628211 & 0xffffffffffffffff
    marker = work/f"req_{key:016x}.c67"
    for suffix, part in parts.items():
        marker.with_suffix(suffix).write_bytes(part)
    return marker


def main(exe, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        root_src = work/"root.hlsl"
        root_src.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8)),DescriptorTable(UAV(u8,numDescriptors=8))"')
        root_file = work/"root.bin"
        subprocess.run([Path(sdk)/"dxc.exe","-T","rootsig_1_1","-E","SIG","-Fo",root_file,root_src],check=True)
        root = root_file.read_bytes()
        # s20:21 is table 0. Load two descriptors into s32:35 and s36:39.
        mov = lambda dst, src: 0xbe800300 | dst<<16 | src
        descriptor = lambda dst, off: [0xf4080000 | dst<<6 | 10, 0xfa000000 | off]
        store = lambda dst: [0xe2102000, 0x80000000 | (dst//4)<<16 | 10<<8 | 8]
        words = [mov(20,4), mov(21,128), *descriptor(32,96), *descriptor(36,32),
                 0x8718ff06,0xfc0, 0xd700000a,0x10080, 0xd7650007,193|(24<<9),
                 0xd7660008,193|((256+7)<<9),0xbf8cc07f,*store(32),*store(36),0xbf810000]

        def run(marker, name, expected=0):
            p=subprocess.run([exe,sdk,work/"cache","gfx1013-group-clear@0",marker,work/name],
                             text=True,capture_output=True)
            if p.returncode != expected: raise AssertionError((p.stdout,p.stderr))
            return p.stdout+p.stderr

        marker=write_request(work,words,root)
        cold=run(marker,"clear.cso")
        assert "u0 parameter=0 byte_offset=32" in cold and "u1 parameter=0 byte_offset=96" in cold
        assert "cache_hit_ms" in run(marker,"warm.cso")
        assert (work/"clear.cso").read_bytes()==(work/"warm.cso").read_bytes()
        marker.with_suffix(".data").write_bytes(b"corrupted")
        assert "key mismatch" in run(marker,"damaged.cso",1)
        # A partial SGPR write breaks the second descriptor. The first store must
        # not cause publication of a partial shader.
        broken=words[:-3]+[mov(37,128)]+words[-3:]
        assert "unsupported" in run(write_request(work,broken,root),"partial.cso",1)
        assert not (work/"partial.cso").exists()
    print("two-table request, register renaming, store order, cache, tamper and partial descriptor passed")


if __name__=="__main__":
    main(*sys.argv[1:])
