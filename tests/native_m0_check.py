"""Numeric M0 in the direct compute path (RDNA2 scalar encoding 124).

The per-wave value must survive overwrites, SCC control flow, CFG joins and
loops. Two complete physical wave32 groups use one guest lane for stores.
M0 memory addressing is deliberately separate from #3172.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request
from native_scalar_lds_check import scalar, sop1, vmove

M0 = 124
SENTINEL = 0x69696969
WORDS = 2064
PREFIX = [sop1(3, 20, 4), sop1(3, 21, 128), 0xf408080a, 0xfa000020,
          scalar(0x1e, 7, 5, 134), vmove(8, 7), sop1(3, 126, 129)]

def mov(dst, src):
    return sop1(3, dst, src)

def stores(regs):
    p = [0xbf8cc07f]
    for j, reg in enumerate(regs):
        p += [0xe0701000 | (4*j), 0x80080008 | (reg << 8)]
    return p + [0xbf810000]

def program():
    p = PREFIX + [
        mov(M0, 132), vmove(1, M0),
        mov(M0, 5), scalar(0, M0, M0, 136), vmove(2, M0),
        mov(M0, 193), scalar(0, M0, M0, 129), vmove(3, M0),
        scalar(0x0a, 10, 129, 128), vmove(4, 10),
        0xb0000000 | M0 << 16 | 0xfffe, vmove(5, M0),
        0xbf068005,  # s_cmp_eq_u32 s5, 0
        0xbf850002,  # SCC1 skips the other branch's write + jump
        mov(M0, 162), 0xbf820001, mov(M0, 145),
        vmove(6, M0), mov(11, M0), vmove(7, 11),
        scalar(0x0a, M0, M0, 176), vmove(9, M0),
        mov(M0, 128),
        scalar(0, M0, M0, 129), 0xbf0a837c, 0xbf85fffd,
        vmove(10, M0),
        mov(126, 128), mov(M0, 137), mov(126, 129), vmove(11, M0),
    ]
    return p + stores((1,2,3,4,5,6,7,9,10,11))

def reference():
    expected = [SENTINEL]*WORDS
    for group in range(2):
        start = 8 + group*16
        branch = 17 if group == 0 else 34
        expected[start:start+10] = [4, 8+group, 0, 1, 0xfffffffe, branch,
                                   branch, 17 if group == 0 else 48, 3, 9]
    return expected

def controls(cli, sdk, work, root):
    def run(label, p, refusal=None):
        marker = request(work, root, p)
        output = work / (label+".cso")
        r = subprocess.run([cli,sdk,work/"cache","gfx1013-direct-cs@0",marker,output],
                           capture_output=True,text=True)
        if refusal is None:
            assert r.returncode == 0 and output.is_file(), (label,r.stdout,r.stderr)
        else:
            assert r.returncode and refusal in r.stderr and not output.exists(), (label,r.stdout,r.stderr)
        print(label, "accepted" if refusal is None else r.stderr.strip(), flush=True)
    run("numeric", program())
    run("undefined", PREFIX+[vmove(1,M0)]+stores((1,)), "unbound numeric M0")
    run("partial_join", PREFIX+[0xbf068005,0xbf850001,mov(M0,132),vmove(1,M0)]+stores((1,)),
        "unbound numeric M0")
    run("descriptor", PREFIX+[mov(M0,32),vmove(1,M0)]+stores((1,)), "descriptor used as mask")
    run("pair_source", PREFIX+[mov(M0,132),sop1(4,10,M0),vmove(1,10)]+stores((1,)),
        "unsupported wave64 scalar pair")
    run("pair_destination", PREFIX+[sop1(4,M0,128),vmove(1,M0)]+stores((1,)),
        "direct scalar move destination")
    run("memory_offset", PREFIX+[mov(M0,132),vmove(1,129),0xe0701000,0x7c080108,0xbf810000],
        "direct DXIL operand encoding")
    return 0

def hardware(replay,sdk,work,root):
    marker = request(work,root,program())
    out = work/"out.bin"
    r = subprocess.run([replay,sdk,marker,work/"cache","0",out,"--direct"])
    if r.returncode == 77:
        return 77
    r.check_returncode()
    actual = struct.unpack(f"<{WORDS}I",out.read_bytes())
    expected = reference()
    assert list(actual) == expected, [(i,hex(a),hex(b)) for i,(a,b) in enumerate(zip(actual,expected)) if a!=b][:16]
    print("M0: two groups, 20 value words and all 2044 guard/untouched words match",flush=True)
    return 0

def main(mode,tool,sdk):
    with tempfile.TemporaryDirectory() as temp:
        work=Path(temp)
        src=work/"root.hlsl"
        src.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root=work/"root.bin"
        subprocess.run([Path(sdk)/"dxc.exe","-T","rootsig_1_1","-E","SIG","-Fo",root,src],check=True)
        return {"controls":controls,"hardware":hardware}[mode](tool,sdk,work,root.read_bytes())

if __name__=="__main__":
    sys.exit(main(*sys.argv[1:]))
