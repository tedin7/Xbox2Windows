"""Check whole-wave zero and pixel coverage using source-built guest shaders."""
import re
import struct
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from native_direct_container_check import patched_pixel, signature
from native_scalar_lds_check import sop1, vmove


def packed(lo, hi, narrow=False):
    move = 3 if narrow else 4
    wqm = 9 if narrow else 10
    # Both paths are scalar until the original coverage is restored.
    # Observe the actual expanded words and which direction EXECZ took.
    # The harness supplies zero in b0. The front end and DXIL optimizer
    # must still evaluate both words from runtime data at draw time.
    def xor_root_word(dst, root):
        return 0x80000000 | (0x12 << 23) | (dst << 16) | (root << 8) | dst
    seq = [sop1(move,20,126), sop1(3,4,255),lo, sop1(3,5,255),hi,
           xor_root_word(4,0), xor_root_word(5,1),
           sop1(3,10,128), sop1(3,11,128), sop1(3,12,129),
           sop1(wqm,126,4),0xbf880002,sop1(move,10,126),sop1(3,12,130),
           sop1(move,126,20)]
    # Byte-sized values remain exactly representable in the float target.
    seq += [vmove(0,10),vmove(1,11),vmove(2,12),vmove(3,10)]
    for v in (0,1,3):
        seq += [(0x16<<25) | (v<<17) | (v<<9) | (152 if v==3 else 128),
                (0x1b<<25) | (v<<17) | (v<<9) | 255,255]
    for v in range(4):
        seq += [0x7e000c00 | (v<<17) | (256+v)]
    return patched_pixel(len(seq),seq,constants=(0,2),wave32=narrow,export=(0xf800180f,0x03020100),
                         input_signature=((b"OSG1",signature(b"SV_Target",64)),))


def raster_program(limit=None, *, lower=None, cross=False):
    # Drop coverage where limit < SV_Position.x. In a surviving quad
    # the horizontal peer's coordinate must remain available, even if its
    # own coverage was dropped. An empty quad branches to EXP_NULL.
    seq=[sop1(4,12,126), sop1(10,126,126)]
    attributes=()
    if limit is None:
        # Perspective-center P1/P2 of attr3.z, followed by 0 > v5.
        p1=0xc8000000 | (5<<18) | (3<<10) | (2<<8)
        seq += [p1, p1 | (1<<16) | 1, 0x7c080000 | (5<<9) | 128]
        attributes=((6,0,15,3,0,0,0,0),)
        inputs, input_address, value = signature(b"TEXCOORD"), 2, 5
    else:
        threshold=struct.unpack("<I",struct.pack("<f",limit))[0]
        seq += [0x7c0200ff,threshold]
        inputs, input_address, value = signature(b"SV_Position",1), 1<<8, 0
    branches=[]
    def filter_and_test():
        seq.extend([0x8a8c6a0c, sop1(10,126,12)])
        branches.append(len(seq))
        seq.append(0)
    filter_and_test()
    if lower is not None:
        assert limit is not None
        seq += [0x7c0800ff,struct.unpack("<I",struct.pack("<f",lower))[0]]
        filter_and_test()
    seq += [vmove(2,250), 0xff040000 | ((0x141 if cross else 0xb1)<<8) | value,sop1(4,126,12)]
    if limit is None:
        # The interpolator may round the fractional input by one ULP. Move
        # it onto a coarser float grid before the bit-exact color comparison.
        seq += [(3<<25) | (2<<17) | (2<<9) | 255, 0x41800000]
    for branch in branches:
        seq[branch] = 0xbf880000 | (len(seq)+3-(branch+1))  # after color export and END
    return patched_pixel(len(seq),seq,input_address=input_address,attributes=attributes,
                         input_signature=((b"ISG1",inputs),(b"OSG1",signature(b"SV_Target",64))),
                         export=(0xf800180f,0x02020202),tail=(0xf8001890,0x80808080,0xbf810000))


def quad_mask(word):
    return sum(15<<shift for shift in range(0,32,4) if word & (15<<shift))


def check_packed_and_exact_exec(draw, compile_guest):
    target = (b"OSG1", signature(b"SV_Target", 64))
    position = (b"ISG1", signature(b"SV_Position", 1))

    def packed_color(code, **kwargs):
        return patched_pixel(len(code), code, export=(0xf8001c0f, 0x05050505), **kwargs)

    def fma(op, src):
        # Destination v5, operands (src, inline 1.0, inline 0).
        return [op | 5, src | (242 << 9) | (128 << 18)]

    low, high = fma(0xcc210000, 242), fma(0xcc220000, 244)
    def interleaved(src):
        return [0xd54b000c, src | (242 << 9) | (128 << 18)]  # v_fma_f32 v12,src,1,0

    expected = [(1.0, 2.0, 1.0, 2.0)] * 16
    for code in (low + high, high + low, low + interleaved(242) + high):
        draw(packed_color(code, input_signature=(target,)), expected)
    for code in (low, low + low, low + interleaved(261) + high, low + fma(0xcc220000, 261)):
        result = compile_guest(packed_color(code, input_signature=(target,)))
        assert result.returncode and "entry VGPR 5" in result.stderr, result.stderr

    # WQM -> exact -> WQM: the exact write to unallocated v4 has no old
    # helper value. Packed arithmetic observes only the eventual color.
    prefix = [sop1(4, 10, 126), sop1(10, 126, 126), sop1(4, 12, 126),
              sop1(4, 126, 10), vmove(4, 242), sop1(4, 126, 12)]
    tail = fma(0xcc210000, 260) + high + [sop1(4, 126, 10)]
    kwargs = dict(input_address=1 << 8, input_signature=(position, target))
    draw(packed_color(prefix + tail, **kwargs), expected)
    quad_read = [vmove(2, 250), 0xff04b104]  # DPP reads v4 from another quad lane
    result = compile_guest(packed_color(prefix + quad_read + tail, **kwargs))
    assert result.returncode and "entry VGPR 4" in result.stderr, result.stderr

    # CMPX compares 4.0 < SV_Position.x. Its read-only arm updates v2
    # only on the right half of the draw; all pixels export after restore.
    def cmpx(body):
        code = [vmove(2, 242), sop1(4, 10, 126), 0x7c2200f6,
                0xbf880000 | len(body), *body, sop1(4, 126, 10)]
        return patched_pixel(len(code), code, export=(0xf800180f, 0x02020202), **kwargs)

    draw(cmpx([vmove(2, 244)]), [(1.0 + (lane % 8 >= 4),) * 4 for lane in range(16)])
    result = compile_guest(cmpx(quad_read))
    assert result.returncode and "graphics branch on a wave mask" in result.stderr, result.stderr


# Cross-quad EXECZ (#2907). A region term (axis, "le"|"ge", value) keeps the lanes with
# SV_Position.axis <= value or >= value; the compare sets VCC for the lanes it removes.
def term_compare(axis, op, value):
    return [(0x7c020000 if op=="le" else 0x7c080000) | ({"x":0,"y":1}[axis]<<9) | 255,
            struct.unpack("<I",struct.pack("<f",value))[0]]


def cross_quad_program(limit, terms, *, base="survivors", region=(sop1(3,14,130),)):
    # Exact coverage drops pixels where limit < x. The EXECZ region runs under
    # WQM of a mask taken from the survivors or from the whole-quad EXEC and
    # narrowed by the terms, and writes only a scalar. After exact coverage is
    # restored every survivor exports that scalar, so a quad with no bit in the
    # mask still observes whether the wave took the branch.
    seq=[sop1(4,12,126), sop1(10,126,126), *term_compare("x","le",limit), 0x8a8c6a0c,  # s[12:13] = survivors
         sop1(3,14,129)]                                                              # s14 = 1
    for k,term in enumerate(terms):
        source=16 if k else (12 if base=="survivors" else 126)
        seq += [*term_compare(*term), 0x8a906a00 | source]                            # s[16:17] = source & ~VCC
    seq += [sop1(10,126,16)]                                                          # exec = WQM(s[16:17])
    branch=len(seq)
    seq += [0, *region]                                    # s_cbranch_execz to the restore; s14 = 2
    seq[branch] = 0xbf880000 | (len(seq)-(branch+1))
    seq += [sop1(4,126,12), vmove(1,14), 0x7e000c00 | (1<<17) | 257]
    return patched_pixel(len(seq),seq,input_address=3<<8,
                         input_signature=((b"ISG1",signature(b"SV_Position",1)),(b"OSG1",signature(b"SV_Target",64))),
                         export=(0xf800180f,0x01010000))


def position(lane):
    return (lane%8+0.5, lane//8+0.5)


def cross_quad_reference(limit, terms, quads, vote, *, base="survivors"):
    """Pixels of cross_quad_program: (x, x, s14, s14) for survivors, the clear value elsewhere.

    `quads` holds the measured desktop quads of the draw: (wave, ((lane, helper), ...)). vote="wave" is the
    lowering: EXEC at the branch is the WQM expansion of the mask, so a quad votes when any of its lanes, helpers
    included, holds a mask bit, and the branch is taken once per measured wave. The negative controls:
    vote="plain" sees only non-helper lanes (a wave vote without the quad OR), vote="survivors" only surviving
    pixels, and vote="quad" lets each quad decide alone."""
    keep=lambda lane: all((position(lane)[a=="y"]<=v) if op=="le" else (position(lane)[a=="y"]>=v)
                          for a,op,v in terms)
    survives=lambda lane, helper: not helper and position(lane)[0]<=limit
    def bit(lane, helper):
        if base=="survivors" and not survives(lane,helper):
            return False
        if vote=="plain" and helper or vote=="survivors" and not survives(lane,helper):
            return False
        return keep(lane)
    voted={(w if vote!="quad" else (w,lanes)) for w,lanes in quads if any(bit(*l) for l in lanes)}
    pixels=[(-1.0,)*4]*16
    for w,lanes in quads:
        flag=2.0 if (w if vote!="quad" else (w,lanes)) in voted else 1.0
        for lane,helper in lanes:
            if survives(lane,helper):
                pixels[lane]=(position(lane)[0],position(lane)[0],flag,flag)
    return pixels


def desktop_quads(work, raster, sdk, backend, vs):
    """Measure the quads and waves of the 8x2 draw with HLSL pixel stages on the same draw.

    Wave operations leave helper lanes out, so each covered pixel reports its wave (the smallest covered id in
    it), the ids of its three quad peers and whether each peer is a helper (IsHelperLane, SM 6.6)."""
    reads={"ids": ("ps_6_0","float4(WaveActiveMin(id),QuadReadAcrossX(id),QuadReadAcrossY(id),"
                            "QuadReadAcrossDiagonal(id))"),
           "helpers": ("ps_6_6","float4(WaveGetLaneCount(),QuadReadAcrossX(h),QuadReadAcrossY(h),"
                                "QuadReadAcrossDiagonal(h))")}
    data={}
    for key,(profile,value) in reads.items():
        source, ps, out = work/f"{key}.hlsl", work/f"{key}.cso", work/f"{key}.bin"
        source.write_text("float4 main(float4 p:SV_Position):SV_Target {uint id=uint(p.y)*8u+uint(p.x);"
                          f"uint h=IsHelperLane()?1u:0u; return {value};}}" if profile=="ps_6_6" else
                          "float4 main(float4 p:SV_Position):SV_Target {uint id=uint(p.y)*8u+uint(p.x);"
                          f"return {value};}}")
        subprocess.run([Path(sdk)/"dxc.exe","-T",profile,"-Fo",ps,source],check=True)
        result=subprocess.run([raster,backend,"--startup-pixel-dump",vs,ps,"8","2",out],capture_output=True,text=True)
        if result.returncode==77: sys.exit(77)
        assert result.returncode==0,(result.stdout,result.stderr)
        values=struct.unpack("<64f",out.read_bytes())
        data[key]=[values[4*i:4*i+4] for i in range(16)]
    quads, widths = {}, set()
    for lane in range(16):
        if data["ids"][lane]==(-1.0,)*4:
            continue  # not covered: the clear value
        wave,*peers=(int(v) for v in data["ids"][lane])
        width,*helpers=(int(v) for v in data["helpers"][lane])
        widths.add(width)
        members=((lane,False),*((p,bool(h)) for p,h in zip(peers,helpers)))
        assert len({m[0] for m in members})==4 and all(position(p)[0]//2==position(lane)[0]//2 and
                                                       position(p)[1]//2==position(lane)[1]//2 for p,_ in members),\
            ("a desktop quad is not the aligned 2x2 block",lane,members)
        quads.setdefault((wave,frozenset(members)),set()).add(lane)
    assert len(widths)==1,widths
    for (wave,members),lanes in quads.items():
        # Every covered lane of a quad reports the same quad, and its smallest covered lane names a wave.
        assert lanes=={l for l,h in members if not h},(wave,members,lanes)
    return [(w,tuple(sorted(m))) for w,m in quads], widths.pop()


VERTEX={"full": "float4 main(uint i:SV_VertexID):SV_Position {return float4(i==1?3:-1,i==2?3:-1,0,1);}",
        # Covers the pixels with x + y < 5.5, so the quad at x = 4..5 has one covered lane, (4,0).
        "diagonal": "float4 main(uint i:SV_VertexID):SV_Position {"
                    "return float4(i==0?-0.25:(i==1?0.5:-5),i==0?-1.5:(i==1?1.5:0),0,1);}"}
# (vertex, limit, terms, base): one quad's survivors hold the mask; a wider mask; an empty wave; a mask in
# covered quads without survivors (WQM before the filter); the same from the survivors; and masks that only
# the diagonal, horizontal or vertical helper of the one covered lane (4,0) holds.
CASES=(("full",6.5,(("x","le",1.0),),"survivors"), ("full",9,(("x","le",4.5),),"survivors"),
       ("full",6.5,(("x","le",-1.0),),"survivors"), ("full",3.5,(("x","ge",4.5),),"wqm"),
       ("full",3.5,(("x","ge",4.5),),"survivors"),
       ("diagonal",4.5,(("x","ge",5.0),("y","ge",1.0)),"wqm"),
       ("diagonal",4.5,(("x","ge",5.0),("y","le",1.0)),"wqm"),
       ("diagonal",4.5,(("x","ge",4.0),("x","le",5.0),("y","ge",1.0)),"wqm"))


def ir_main(dump):
    """Without a GPU: the vote's three quad reads and the wave vote come before any conditional branch, where
    every lane of the quad, helpers included, still runs them (quad operations need quad-uniform control flow).

    The patterns follow the frontend's current text form (immediate i8 operands, attributes written inline on
    the declaration); a change to that form, for example attribute groups, needs these patterns updated."""
    with TemporaryDirectory() as tmp:
        blob=Path(tmp)/"cross.bin"
        blob.write_bytes(cross_quad_program(6.5,(("x","le",1.0),)))
        result=subprocess.run([dump,blob],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        body=result.stdout[result.stdout.index("define void @main"):]
        first_branch=body.index("\n  br i1 ")  # the first conditional branch: the vote's own
        quad=[m.group(1) for m in re.finditer(r"@dx\.op\.quadOp\.i32\(i32 123, i32 %\w+, i8 (\d)\)",body)]
        vote=[m.start() for m in re.finditer(r"@dx\.op\.waveAnyTrue\(i32 113, i1 %\w+\)",body)]
        assert sorted(quad)==["0","1","2"] and len(vote)==1,(quad,vote)
        assert all(m.start()<first_branch for m in re.finditer(r"call \w+ @dx\.op\.(quadOp|waveAnyTrue)",body)),body
        # The same attributes as DXC gives the HLSL intrinsics.
        for declaration in ("declare i32 @dx.op.quadOp.i32(i32, i32, i8) nounwind\n",
                            "declare i1 @dx.op.waveAnyTrue(i32, i1) nounwind\n"):
            assert declaration in result.stdout,(declaration,result.stdout)
        print("cross-quad EXECZ IR: QuadReadAcrossX/Y/Diagonal and WaveActiveAnyTrue before the first branch")


def main(cli, raster, sdk, backend="hardware"):
    with TemporaryDirectory() as tmp:
        work=Path(tmp)
        source, vs, ps = work/"source", work/"vs.cso", work/"ps.cso"
        def vertex(kind):
            source.write_text(VERTEX[kind])
            subprocess.run([Path(sdk)/"dxc.exe","-T","vs_6_6","-Fo",vs,source],check=True)
        vertex("full")
        def compile_guest(blob):
            source.write_bytes(blob)
            return subprocess.run([cli,sdk,work/"cache","gfx1013-container-ps",source,ps],
                                  capture_output=True,text=True)
        def run(blob, pixels):
            result=compile_guest(blob)
            assert not result.returncode,result.stderr
            expected=work/"expected.bin"
            expected.write_bytes(b"".join(struct.pack("<4f",*pixel) for pixel in pixels))
            result=subprocess.run([raster,backend,"--startup-pixel-words",vs,ps,"8","2",expected],
                                  capture_output=True,text=True)
            if result.returncode==77: sys.exit(77)
            return result
        def draw(blob, pixels):
            result=run(blob,pixels)
            assert result.returncode==0,(result.stdout,result.stderr)
            print(result.stdout.strip())
        # On a 2x2 quad, compare the vertical neighbor with the negated
        # current value. The equal case on column two checks ordered GT.
        cmp_code=[sop1(10,126,126),0x06220300,0x062222ff,0xc0600000, # v17 = x+y-3.5
                  0x7c0822fa,0xff4c4e11,0x7e0402f2,0x02000480]
        def quad_compare(code):
            return patched_pixel(len(code),code,input_address=0x300,
                                 input_signature=((b"ISG1",signature(b"SV_Position",1)),))
        expected=[(float(x>=3),)*4 for y in range(2) for x in range(8)]
        draw(quad_compare(cmp_code),expected)
        # The high word of v54 contains floor(x). Its low word is zero.
        # -1 as 32-bit raw bits is qNaN (class bit 1), while 0x7c01 is
        # binary16 sNaN (bit 0). Both arms write v0, on complementary EXEC.
        for half in (False,True):
            selector=[0x7d1e6cf9,0x0586b40f] if half else [0x7d106cf9,0x0586b4c1]
            seq=[0x7e6c0f00,0x346c6c90,0xbe8f03ff,0x7c01,0xbebc047e,
                 *selector,0xbeb40834,0x87fe7e34,0xbf880001,0x7e0002f2,
                 0x8afe7e3c,0x7e0002f0,0xbefe043c]
            flags=1 if half else 2
            reference=[(0.5 if x & flags else 1.0,)*4 for y in range(2) for x in range(8)]
            draw(quad_compare(seq),reference)
        for control in (0xff4ce411,0xff0c4e11): # Identity permutation or unnegated src1.
            wrong=cmp_code.copy()
            wrong[5]=control
            failed=run(quad_compare(wrong),expected)
            assert failed.returncode==1 and "pixels differ" in failed.stderr,failed.stderr
        check_packed_and_exact_exec(draw, compile_guest)
        for lo,hi,narrow in ((0,0,False),(0,1,False),(0x80000000,0,False),
                             (0x80000021,0x42,False),(0,1,True),(0x22,0,True)):
            a,b=quad_mask(lo),0 if narrow else quad_mask(hi)
            pixel=(a&255,b&255,2 if a or b else 1,a>>24)
            draw(packed(lo,hi,narrow),[pixel]*16)
            print(f"wave{32 if narrow else 64} {hi:08x}:{lo:08x} -> {pixel}")
        for limit in (0,0.5,2.5,9):
            values=[(float((x^1)+0.5) if x+0.5<=limit else -1.0,)*4 for y in range(2) for x in range(8)]
            draw(raster_program(limit),values)
            print(f"filtered coverage limit={limit}: {sum(x[0]!=-1.0 for x in values)}/16 pixels survived")
        for lower in (1.5,4.5):
            values=[(float((x^1)+0.5) if lower<=x+0.5<=2.5 else -1.0,)*4 for y in range(2) for x in range(8)]
            draw(raster_program(2.5,lower=lower),values)
            print(f"two coverage filters [{lower},2.5]: {sum(x[0]!=-1.0 for x in values)}/16 pixels survived")
        rejected=compile_guest(raster_program(2.5,cross=True))
        assert rejected.returncode and "complete guest vote or a quad-local continuation" in rejected.stderr,rejected.stderr
        # Cross-quad EXECZ: the CPU reference comes from the measured desktop quads and waves of each draw.
        controls, cross_controls, measured = {}, 0, {}
        for kind,limit,terms,base in CASES:
            vertex(kind)
            if kind not in measured:
                measured[kind]=desktop_quads(work,raster,sdk,backend,vs)
                quads,width=measured[kind]
                waves={w for w,_ in quads}
                print(f"{kind} draw: {width} lanes, {len(quads)} quads in {len(waves)} waves, "
                      f"{sum(1 for w in waves if sum(1 for q,_ in quads if q==w)>1)} waves with more than one quad, "
                      f"{sum(h for _,lanes in quads for _,h in lanes)} helper lanes")
            quads,_=measured[kind]
            blob=cross_quad_program(limit,terms,base=base)
            reference=lambda vote: cross_quad_reference(limit,terms,quads,vote,base=base)
            whole=reference("wave")
            draw(blob,whole)
            label=f"{kind} limit={limit} {base} {terms}"
            print(f"cross-quad {label}: {sum(p[0]!=-1.0 for p in whole)}/16 survive, "
                  f"s14 {sorted({p[2] for p in whole if p[0]!=-1.0})}")
            for vote in ("quad","plain","survivors"):
                other=reference(vote)
                if other==whole:
                    continue
                result=run(blob,other)
                assert result.returncode==1 and "pixels differ" in result.stderr,(vote,label,result.stderr)
                print(f"  {vote} vote reference: {next(l for l in result.stderr.splitlines() if 'pixels differ' in l)}")
                controls[vote]=controls.get(vote,0)+1
                cross_controls+=vote=="quad"
        assert cross_quad_reference(6.5,(("x","le",-1.0),),measured["full"][0],"wave")== \
               cross_quad_reference(6.5,(("x","le",-1.0),),measured["full"][0],"quad")
        assert controls.get("plain",0)>=3,controls  # the three single-helper masks are decided by the quad OR
        vertex("full")
        # Lane reads, vector writes, branches and EXEC changes inside the region keep a named refusal.
        for label,region,reason in (
                ("DPP inside the region",(0x7e0402fa,0xff04b100),"may only write scalars"),
                ("VGPR write inside the region",(vmove(1,14),),"may only write scalars"),
                ("EXEC read inside the region",(sop1(4,20,126),),"may only write scalars"),
                ("branch inside the region",(0xbf820000,),"region contains a branch"),
                ("EXEC rewritten inside the region",(sop1(4,126,16),),"must branch to the exact coverage restore"),
                ("s_setreg_b32 inside the region",(0xb98e0801,),"hardware register"),
                ("s_getreg_b32 inside the region",(0xb9140801,),"hardware register")):
            rejected=compile_guest(cross_quad_program(6.5,(("x","le",1.0),),region=region))
            assert rejected.returncode and reason in rejected.stderr,(label,rejected.stderr)
            print(f"refused: {label}: ok")
        # The pixel program now consumes an interpolated value produced by
        # the VS, including a negative coordinate from a WQM helper.
        source.write_text("struct Out { float4 tc:TEXCOORD; float4 pos:SV_Position; };"
                          "Out main(uint i:SV_VertexID) { Out o;"
                          "o.pos=float4(i==1?3:-1,i==2?3:-1,0,1);"
                          "o.tc=float4(0,0,3.25-(o.pos.x+1)*4,0); return o; }")
        subprocess.run([Path(sdk)/"dxc.exe","-T","vs_6_6","-Fo",vs,source],check=True)
        values=[(16+3.25-((x^1)+0.5) if x+0.5<=3.25 else -1.0,)*4 for y in range(2) for x in range(8)]
        draw(raster_program(),values)
        print("attr3.z perspective interpolation, VCC filter and WQM/EXECZ: all 16 pixels matched")
        print(f"cross-quad EXECZ negative controls that differ from the readback: {controls}")
        if not cross_controls:
            reason=f"no desktop wave holds two quads ({measured['full'][1]} lanes)"
            if backend=="hardware":
                # Every other check passed; the cross-quad decision itself was not exercised.
                print(f"skip: the cross-quad quad-local control needs two quads in one wave: {reason}")
                sys.exit(77)
            print(f"SKIPPED cross-quad quad-local control: {reason}")


if __name__=="__main__":
    if sys.argv[1]=="ir":
        ir_main(sys.argv[2])
    else:
        main(*sys.argv[1:])
