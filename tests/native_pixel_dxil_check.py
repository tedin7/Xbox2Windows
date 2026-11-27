"""Compare a source-built gfx1013 packet with independently written HLSL."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def main(exe, sdk):
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        def compile_one(profile, source, output, success=True):
            p = subprocess.run([exe, sdk, work/"cache", profile, work/source, work/output],
                               capture_output=True, text=True)
            if (p.returncode == 0) != success:
                raise AssertionError((p.stdout, p.stderr))
            return p.stdout if success else p.stderr

        words = [0x7e0002f2,0x7e0202f0,0x7e040280,0x7e0602ff,0x3e800000,
                 0xf800180f,0x03020100,0xbf810000]
        (work/"pixel.bin").write_bytes(struct.pack("<8I", *words))
        (work/"pixel.hlsl").write_text("float4 main() : SV_Target0 { return float4(1,0.5,0,0.25); }")
        mode = "gfx1013-constant-ps"
        assert "cold_compile" in compile_one(mode,"pixel.bin","native.cso")
        assert "cache_hit" in compile_one(mode,"pixel.bin","warm.cso")
        cached, = (work/"cache").glob("*.bin")
        damaged = bytearray(cached.read_bytes())
        damaged[-1] ^= 1
        cached.write_bytes(damaged)
        assert "cold_compile" in compile_one(mode,"pixel.bin","recovered.cso")
        compile_one("ps_6_0","pixel.hlsl","ref.cso")
        expected=(work/"ref.cso").read_bytes()
        assert all((work/name).read_bytes() == expected for name in ("native.cso","warm.cso","recovered.cso"))
        words[2]=0xbf800000  # s_nop is outside the supported instruction family
        (work/"pixel.bin").write_bytes(struct.pack("<8I", *words))
        assert "instruction at 8" in compile_one(mode,"pixel.bin","invalid.cso",False)
        assert not (work/"invalid.cso").exists()
        sources=[work/f"batch{i}.hlsl" for i in range(4)]
        for i,path in enumerate(sources):
            path.write_text(f"float4 main() : SV_Target0 {{ return {i+1}; }}")
        args=[exe,sdk,work/"parallel-cache","--jobs","3","ps_6_0"]
        pairs=[str(arg) for path in sources for arg in (path,path.with_suffix(".cso"))]
        cold=subprocess.run(args+pairs,capture_output=True,text=True,check=True)
        warm=subprocess.run(args+pairs,capture_output=True,text=True,check=True)
        assert cold.stdout.count("cold_compile")==4 and warm.stdout.count("cache_hit")==4
        for path in sources:
            compile_one("ps_6_0",path.name,"serial.cso")
            assert path.with_suffix(".cso").read_bytes()==(work/"serial.cso").read_bytes()
        bad=pairs.copy()
        bad[-1]=pairs[1].upper()
        assert "distinct" in subprocess.run(args+bad,capture_output=True,text=True).stderr
        # One failed input keeps its previous output. Other jobs still finish.
        previous=sources[0].with_suffix(".cso").read_bytes()
        sources[0].write_text("invalid HLSL")
        result=subprocess.run(args+pairs,capture_output=True,text=True)
        assert result.returncode==1 and "job=1" in result.stderr and result.stdout.count("cache_hit")==3
        assert sources[0].with_suffix(".cso").read_bytes()==previous
        for pragma,error in [("x2w_vop2 340804ff","literal length"),
                             ("x2w_vop2 02060501","unsupported VOP2 opcode"),
                             ("x2w_vop3 00000000","unknown native compute pragma")]:
            (work/"bad-ir.hlsl").write_text(f"#pragma {pragma}\n[numthreads(1,1,1)] void main() {{}}")
            assert error in compile_one("cs_6_0","bad-ir.hlsl","invalid-ir.cso",False)
            assert not (work/"invalid-ir.cso").exists()
    print("native pixel DXIL, independent reference, persistent hit, cache repair and refusal passed")


if __name__ == "__main__":
    main(*sys.argv[1:])
