"""Reproduce BCPack format constants with chosen synthetic lane streams.

SPDX-License-Identifier: MIT

The GDK is a development oracle only. No DLL bytes or memory tables are read here.
BC6H wire bit descriptors are parsed from the public MIT DirectXTex source.
"""
import ctypes as c
import json
import argparse
import hashlib
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sdk", type=Path, required=True, help="Installed BCPack.dll, development oracle only")
parser.add_argument("--bridge", type=Path, required=True, help="Source-built bcpack_oracle_bridge.dll")
parser.add_argument("--bc6-reference", type=Path, required=True, help="Pinned public DirectXTex BC6HBC7.cpp")
parser.add_argument("--output", type=Path, required=True, help="Generated .inl (normally under out/)")
parser.add_argument("--storage-goldens", type=Path, help="Also regenerate the source-built storage fixture's expected blocks")
args = parser.parse_args()
sdk = c.CDLL(str(args.sdk.resolve()))
dec = sdk.BCPackDecompress
dec.argtypes = [c.c_size_t, c.c_void_p, c.c_size_t, c.c_void_p, c.c_size_t]
dec.restype = c.c_int32
encoder = c.CDLL(str(args.bridge.resolve()))
enc = encoder.encode_symbols
enc.argtypes = [c.c_uint, c.POINTER(c.c_uint16), c.c_size_t, c.c_void_p, c.c_size_t]
enc.restype = c.c_size_t
N = {1:12, 2:15, 3:12, 4:12, 5:12, 6:16, 7:16}
PER = {b:1 if b in (1,4) else 2 for b in N}
LINE = [0, 3, 1, 2]
BC6_BITS = [[0]*4,[10,5,5,5],[7,6,6,6],[11,5,4,4],[11,4,5,4],
            [11,4,4,5],[9,5,5,5],[8,6,5,5],[8,5,6,5],[8,5,5,6],
            [6,6,6,6],[10,10,10,10],[11,9,9,9],[12,8,8,8],[16,4,4,4]]


def wire_descriptors():
    source = args.bc6_reference.read_bytes()
    if hashlib.sha256(source).hexdigest() != "187434fab12351df0f875ae09390981034e8479159a6f4aaaf0904f007182b8a":
        raise ValueError("DirectXTex reference does not match pinned a8e5dbf42909aeeb9fcf9c2ae399ae3c06737379")
    text = source.decode("utf-8-sig")
    text = text.split('const D3DX_BC6H::ModeDescriptor D3DX_BC6H::ms_aDesc',1)[1]
    text = text.split('// Mode, Partitions, Transformed, IndexPrec',1)[0]
    desc = [[]]
    for mode, body in re.findall(r'\{\s*// Mode (\d+)[^\n]*\n(.*?)\n\s*\},',text,re.S):
        mode = int(mode)
        pairs = [(f,int(b)) for f,b in re.findall(r'\{\s*(\w+),\s*(\d+)\s*\}',body)]
        assert len(pairs) == 82 and mode == len(desc)
        # BCPack's packer writes the high base bits ascending in modes 13/14.
        if mode in (13,14):
            for f in ['RW','GW','BW']:
                positions = [i for i,(field,bit) in enumerate(pairs) if field == f and bit >= 10]
                for i,b in zip(positions,sorted(pairs[i][1] for i in positions)):
                    pairs[i] = (f,b)
        desc.append(pairs[:82 if mode <= 10 else 65])
    assert len(desc) == 15
    return desc


DESC = wire_descriptors()


def interleave(a,b):
    return sum(((a>>k&1)<<(2*k)) | ((b>>k&1)<<(2*k+1)) for k in range(4))


def split(v):
    return [sum((v>>(2*k+j)&1)<<k for k in range(4)) for j in range(2)]


def query(fmt, ep=0, ix=0, mode=11, rank=0):
    per = PER[fmt]
    s = [0] * (N[fmt] * per)
    delta = (ep - ix + 128) % 256 - 128
    zz = 2*delta if delta >= 0 else -2*delta-1
    s[:per] = split(zz) if per == 2 else [zz]
    s[per:2*per] = split(ix) if per == 2 else [ix]
    if fmt == 1:
        s[8:] = [rank]*4
    elif fmt == 2:
        s[26:] = [rank]*4
    elif fmt == 6:
        s[30] = mode-1
    v = (c.c_uint16*len(s))(*s)
    packed = c.create_string_buffer(65536)
    n = enc(fmt,v,per,packed,len(packed))
    assert n, (fmt,ep,ix,mode,rank,'encode')
    size = 8*per
    out = c.create_string_buffer(b'\xa5'*(size+32),size+32)
    hr = dec(fmt,c.addressof(out)+16,size,packed,n)
    assert hr >= 0, (fmt,ep,ix,mode,rank,hex(hr&0xffffffff))
    assert out.raw[:16] == b'\xa5'*16 and out.raw[-16:] == b'\xa5'*16
    return out.raw[16:-16]


def colour(raw):
    a,b = int.from_bytes(raw[:2],'little'),int.from_bytes(raw[2:4],'little')
    ep = [a>>11,a>>5&63,a&31,b>>11,b>>5&63,b&31]
    word = int.from_bytes(raw[4:8],'little')
    ix = [LINE[word>>(2*p)&3] for p in range(16)]
    return ep,ix


def unpack6(raw,mode):
    word = int.from_bytes(raw,'little')
    fields = {f:0 for f in ['RW','RX','RY','RZ','GW','GX','GY','GZ','BW','BX','BY','BZ']}
    for i,(f,bit) in enumerate(DESC[mode]):
        if f in fields: fields[f] |= ((word>>i)&1)<<bit
    ep = [[0]*3 for _ in range(4)]
    for k,e in enumerate('WXYZ'):
        if mode > 10 and k > 1: continue
        for channel,ch in enumerate('RGB'):
            bits = BC6_BITS[mode][0 if k == 0 else channel+1]
            ep[k][channel] = (fields[ch+e] << (16-bits)) ^ 0x8000
    at = 82 if mode <= 10 else 65
    bits = 3 if mode <= 10 else 4
    ix = []
    for p in range(16):
        width = bits - int(p == 0 or mode <= 10 and p == 15)
        value = (word>>at)&((1<<width)-1)
        ix.append(((value-(1<<(bits-1))) << (16-bits))&65535)
        at += width
    assert at == 128
    return ep,ix


def main():
    tables = {'method':'Synthetic BCPack lane streams -> decoder API -> public BC block fields; no SDK binary-table extraction',
              'bc6_wire':DESC, 'bc6_bits':BC6_BITS}
    for fmt in (1,2):
        ranks=[]
        for rank in range(256):
            _, ix = colour(query(fmt,rank=rank)[0 if fmt == 1 else 8:])
            ranks.append([ix[p] for p in (0,1,4,5)])
        zero = ranks.index([0]*4)
        rows=[]
        for r in range(1,256):
            raw = query(fmt,ep=r,ix=r,rank=zero)
            ep,ix=colour(raw[0 if fmt == 1 else 8:])
            row={'ep':ep,'ix':ix}
            if fmt == 2:
                a=int.from_bytes(raw[:8],'little')
                row['alpha']=[((a>>(4*p))&15)^8 for p in range(16)]
            rows.append(row)
        tables[f'bc{fmt}_ranks']=ranks
        tables[f'bc{fmt}_dictionary']=rows
        print(f'BC{fmt}: {len(ranks)} quadrant ranks and {len(rows)} predictor entries recovered',flush=True)
    rows=[]
    for r in range(1,256):
        a,_=unpack6(query(6,ep=r,mode=14),14)
        b,_=unpack6(query(6,ep=r,mode=11),11)
        c,_=unpack6(query(6,ep=r,mode=2),2)
        _,ix=unpack6(query(6,ix=r,mode=11),11)
        rows.append({'ep':[a[0],b[1],c[2],c[3]],'ix':ix})
    tables['bc6_dictionary']=rows
    write_tables(tables, args.output)
    if args.storage_goldens:
        storage_goldens(args.storage_goldens)
    print(f'BC6H: {len(rows)} predictor entries recovered',flush=True)


def initializer(values):
    return '{ ' + ', '.join(initializer(v) if isinstance(v,list) else str(v) for v in values) + ' }'


def storage_goldens(out):
    build = encoder.storage_symbols
    build.argtypes = [c.c_uint, c.POINTER(c.c_uint16), c.c_size_t]
    build.restype = c.c_size_t
    lines = ["// GDK 1.2603.3.0 golden BC blocks from random_symbols(fmt,4,0x7700+fmt).",
             "// Entirely synthetic input, no SDK or title bytes. SPDX-License-Identifier: MIT", "#pragma once"]
    for fmt in (2, 6):
        symbols = (c.c_uint16 * 128)()
        assert build(fmt, symbols, 128)
        packed = c.create_string_buffer(65536)
        size = enc(fmt, symbols, 8, packed, len(packed))
        assert size
        output = c.create_string_buffer(64)
        assert dec(fmt, output, 64, packed, size) >= 0
        lines.append(f"inline constexpr unsigned char BC{fmt}_STORAGE_EXPECTED[64] = {{")
        for at in range(0, 64, 16):
            lines.append("    " + ", ".join(f"0x{v:02x}" for v in output.raw[at:at+16]) + ",")
        lines.append("};")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_tables(t, out):
    lines=['// BCPack format constants reconstructed with synthetic lane streams.',
           '// No SDK binaries or title data are embedded. See docs/experiments/bcpack-standalone.md.',
           '// BC6H wire descriptors adapt the MIT-licensed Microsoft DirectXTex BC6HBC7.cpp.',
           '// Copyright (c) Microsoft Corporation. All rights reserved.',
           '// SPDX-License-Identifier: MIT', '',
           'struct Bc1FullState { uint8_t ep[6]; uint8_t ix[16]; };',
           'struct Bc2State { uint8_t alpha[16]; uint8_t ep[6]; uint8_t ix[16]; };',
           'struct Bc6State { uint16_t ep[4][3]; uint16_t ix[16]; };',
           'struct Bc6WireBit { uint8_t field; uint8_t bit; };','']
    for fmt in (1,2):
        lines.append(f'inline constexpr uint8_t BC{fmt}_POSITION_RANK[256][4] = {{')
        lines.extend('    '+initializer(row)+',' for row in t[f'bc{fmt}_ranks'])
        lines.extend(['};',''])
        typ='Bc1FullState' if fmt==1 else 'Bc2State'
        lines.append(f'inline constexpr {typ} BC{fmt}_FULL_DICTIONARY[255] = {{')
        fields=['ep','ix'] if fmt==1 else ['alpha','ep','ix']
        lines.extend('    '+initializer([row[f] for f in fields])+',' for row in t[f'bc{fmt}_dictionary'])
        lines.extend(['};',''])
    lines.append('inline constexpr Bc6State BC6_DICTIONARY[255] = {')
    lines.extend('    '+initializer([row['ep'],row['ix']])+',' for row in t['bc6_dictionary'])
    lines.extend(['};','','inline constexpr uint8_t BC6_ENDPOINT_BITS[15][4] = {'])
    lines.extend('    '+initializer(row)+',' for row in t['bc6_bits'])
    lines.extend(['};','','// 0 = mode bit, 1 = partition bit; 2 + 3 * endpoint + channel = endpoint bit.',
                  'inline constexpr Bc6WireBit BC6_WIRE[14][82] = {'])
    for mode,row in enumerate(t['bc6_wire'][1:],1):
        vals=[]
        for f,b in row:
            field = 0 if f in ('M','NA') else 1 if f=='D' else 2+3*'WXYZ'.index(f[1])+'RGB'.index(f[0])
            vals.append([field,b])
        lines.append('    { // Mode '+str(mode))
        for pos in range(0,len(vals),10):
            lines.append('        '+', '.join(initializer(v) for v in vals[pos:pos+10])+',')
        lines.append('    },')
    lines.extend(['};',''])
    out.write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
