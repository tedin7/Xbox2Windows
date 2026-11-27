"""Index source-built native containers and replay their raw runtime form."""
import pathlib
import subprocess
import sys
import tempfile

from native_direct_container_check import patched_pixel, publish, vertex, words


def main(compiler, sdk):
    def run(*args, expected=None):
        result = subprocess.run([compiler, *args], capture_output=True, text=True)
        if expected is None:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode and expected in result.stderr, result.stderr
        return result.stdout

    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        library, cache, requests = (base / s for s in ("library", "cache", "requests"))
        library.mkdir()
        # The old raw reader has no PS state for this scalar branch. Exact
        # native programs recover the full stage ABI from the catalog.
        branch = [0xbf008180, 0xbf850002, 0x7e0002f2, 0xbf820001, 0x7e0002f4]
        prefix = [0xd5800000, 255, 17, 0xd5800000, 255, 23]
        program = prefix + branch
        end = 0xbf810000
        vs = words(0x7e020b00, 0xf80008cf, 0x01010101, end)
        ps = words(*program, 0xf800180f, 0, end)
        pixel_file = library / "pixel.cdxil"
        pixel_file.write_bytes(patched_pixel(len(program), program))
        vertex_file = library / "vertex.cdxil"
        vertex_file.write_bytes(vertex())
        root = b"DXBC" + bytes(16) + words(1,68,1,36) + b"RTS0" + words(24,2,0,24,0,24,0)
        stem = publish(requests, ps, vs, root)
        def batch(folder, output, **kwargs):
            return run(sdk, cache, "gfx1013-container-pair", "--requests", folder, output, **kwargs)
        batch(requests, base / "bare", expected="verified stage interface")
        install = ("--index-graphics-interfaces", library, cache)
        run(*install)
        for name, hit in (("cold",0), ("warm",1)):
            assert f"frontend_cache_hit={hit}" in batch(requests, base / name)
        run(sdk,cache,"gfx1013-container-pair",vertex_file,base/"vertex.cso",pixel_file,base/"pixel.cso")
        for stage, name in (("vs","vertex"),("ps","pixel")):
            binary = (base / (name+".cso")).read_bytes()
            assert all((base / mode / (stem+"."+stage+".cso")).read_bytes() == binary for mode in ("cold","warm"))
        # A second implementation shares the embedded ID. Exact matches
        # still work, while a third unmatched implementation cannot pick one.
        alternate = prefix + [0xbf800000] + branch
        (library/"alternate.cdxil").write_bytes(patched_pixel(len(alternate),alternate))
        run(*install)
        batch(requests,base/"exact")
        other = base/"other"
        publish(other, words(*prefix,0xbf800000,0xbf800000,*branch,0xf800180f,0,end),vs,root)
        batch(other,base/"ambiguous",expected="ambiguous reviewed graphics interfaces")
        # A failed rebuild must leave the published generation usable.
        (library/"broken.cdxil").write_bytes(b"bad")
        assert subprocess.run([compiler,*install],capture_output=True).returncode
        batch(requests,base/"retained")

        # An exact VS writes a four-component image through the second
        # guest slot. The serialized root covers just the first slot.
        # The FSSM offset, descriptor width, class and SGPR establish the
        # binding used by the C++ linker and copied to the runtime cache.
        images = base / "images"
        images.mkdir()
        (images / "pixel.cdxil").write_bytes(pixel_file.read_bytes())
        vs_file = images / "store.cdxil"
        op = [0xbe810380, 0xf40c0200, 0xfa000020,
              0x7e000280, 0x7e020280, 0x7e0802f2, 0x7e0a02f2,
              0xf0200f08, 0x80020400, 0xf80008cf, 0, end]
        root_src, root_bin = base / "uav.hlsl", base / "uav.bin"
        root_src.write_text('#define RS "DescriptorTable(UAV(u0))"')
        subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-T", "rootsig_1_1", "-E", "RS",
                        root_src, "-Fo", root_bin], check=True)
        job = base / "image-request"
        publish(job, ps, words(*op), root_bin.read_bytes())
        for kind, width, sgpr, error in (
                (3, 8, 0, None), (0, 8, 0, "disagrees with reflected table record"),
                (3, 4, 0, "disagrees with reflected table record"),
                (3, 8, 1, "no matching user SGPR")):
            vs_file.write_bytes(vertex(program=op, table=True, uses=[(0x300 | kind, 8, 0, 0, width | sgpr << 8)]))
            run("--index-graphics-interfaces", images, cache)
            output = base / f"store-{kind}-{width}-{sgpr}"
            batch(job, output, expected=error)
            if error:
                assert not list(output.glob("*.cso"))
            else:
                assert "frontend_cache_hit=1" in batch(job, base / "warm-store")


if __name__ == "__main__":
    main(*sys.argv[1:])
