"""Exercise packed CBV/SRV address pairs in the raw compute entry ABI."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_direct_container_check import compute, words


def main(exe, sdk):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        def signature(text):
            src = work / "root.hlsl"
            src.write_text(f'#define RS "{text}"')
            binary = work / "root.bin"
            subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "RS",
                            "-Fo", binary, src], check=True)
            return binary.read_bytes()

        def compile_case(code, root, user=10, error=None):
            sa = struct.pack("<7I", 0, 0, 0, 32 | (1 << 16), 1, 0, user << 1)
            marker = write_request(work, code, root, sa)
            result = subprocess.run([exe, sdk, work / "cache", "gfx1013-direct-cs@0",
                                     marker, marker.with_suffix(".cso")], text=True, capture_output=True)
            if error:
                assert result.returncode and error in result.stderr, result.stderr
                assert not marker.with_suffix(".cso").exists()
            else:
                assert result.returncode == 0, result.stderr
            return result.stdout

        prefix = "RootConstants(num32BitConstants=2,b1),"
        root = signature(prefix + "CBV(b0),SRV(t0)")
        end = 0xbf810000
        # s[6:7] and s[8:9] belong to different root parameters.
        # Copy the second address to s[2:3] before reading at byte 48.
        read_cbv = [0xf40c0403, 0xfa000010]
        copy_srv = 0xbe820408
        read_srv = [0xf40c0601, 0xfa000030]
        code = [*read_cbv, copy_srv, *read_srv, end]
        cold = compile_case(code, root)
        assert "t0 root=1 bytes=0 min_size=48" in cold, cold
        assert "t1 root=2 bytes=0 min_size=80" in cold, cold
        assert "frontend_cache_hit=1" in compile_case(code, root)
        compile_case([*read_cbv, copy_srv, 0xbe830380, *read_srv, end], root,
                     error="mismatched root address high half")
        compile_case(code, root, user=9, error="root descriptor pair exceeds user SGPRs")
        for wrong in ("UAV(u0)", "CBV(b0,visibility=SHADER_VISIBILITY_VERTEX)"):
            compile_case(code, signature(prefix + wrong + ",SRV(t0)"),
                         error="compute root descriptor type or visibility")

        # The first table range occupies slot zero. Its appended range at slot
        # one is either CBV or SRV, although both shaders perform the same SMEM load.
        read_table = [0xbe850380, 0xf4080202, 0xfa000020, 0xf42000c4, 0xfa000000, end]
        for declaration, expected in (("CBV(b1)", "declared-cbv"), ("SRV(t1)", "srv")):
            table = signature("DescriptorTable(SRV(t0)," + declaration + ")")
            assert "table_view=" + expected in compile_case(read_table, table, user=5)

        container_code = words(0xbe810380, 0xf4080200, 0xfa000020, 0xf42000c4, 0xfa000000, end)
        for name, kind, width, sgpr, error in (
                ("cbv", 2, 4, 0, ""), ("srv", 0, 4, 0, ""),
                ("uav", 3, 4, 0, "disagrees with reflected table record"),
                ("image", 0, 8, 0, "disagrees with reflected table record"),
                ("sgpr", 2, 4, 1, "no matching user SGPR")):
            marker = work / (name + ".cdx")
            record = (0x300 | kind, 8, 0, 0, width | sgpr << 8)
            marker.write_bytes(compute(container_code, table=True, uses=[record]))
            args = [exe, sdk, work / "cache", "gfx1013-container-cs", marker, marker.with_suffix(".cso")]
            result = subprocess.run(args, capture_output=True, text=True)
            if error:
                assert result.returncode and error in result.stderr, result.stderr
            else:
                expected = "table_view=" + ("reflected-cbv" if kind == 2 else "srv")
                assert result.returncode == 0 and expected in result.stdout, (result.stdout, result.stderr)
                warm = subprocess.run(args, check=True, capture_output=True, text=True).stdout
                assert "frontend_cache_hit=1" in warm and expected in warm, warm


if __name__ == "__main__":
    main(*sys.argv[1:])
