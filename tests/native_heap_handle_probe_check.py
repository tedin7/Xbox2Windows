"""Cycle 1 of the global SRV work: compile a reference ResourceDescriptorHeap shader with DXC, show how DXC spells the heap handle and
its shader flags, then run the native heap-handle probe with those flags (exit 77 from the probe is a skip)."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REFERENCE = """ByteAddressBuffer selection : register(t0);
RWByteAddressBuffer output : register(u0);
[numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {
    ByteAddressBuffer data = ResourceDescriptorHeap[selection.Load(0)];
    output.Store(id.x * 4, data.Load(id.x * 4));
}
"""


def main(probe, sdk, adapter):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        source = work / "reference.hlsl"
        source.write_text(REFERENCE)
        listing = work / "reference.ll"
        subprocess.run([str(Path(sdk) / "dxc.exe"), "-T", "cs_6_6", "-E", "main", "-Fc", str(listing), str(source)], check=True,
                       capture_output=True)
        text = listing.read_text()
        flags = re.search(r"!\{i32 0, i64 (\d+)", text)
        if not flags:
            raise AssertionError("the reference listing has no shader flags tag")
        for line in text.splitlines():
            if "createHandleFromHeap" in line or "annotateHandle(i32 216" in line and "ResourceProperties { i32 11" in line:
                print("reference:", line.strip())
        print("reference shader flags:", flags.group(1))
        result = subprocess.run([probe, adapter, sdk, str(work / "probe"), flags.group(1)], capture_output=True, text=True)
        print(result.stdout, end="")
        print(result.stderr, end="")
        if result.returncode not in (0, 77):
            # Show how the two containers differ (module, shader flags, pipeline state validation data).
            import difflib
            listings = {}
            for name in ("reference", "native"):
                container = work / "probe" / f"{name}.dxil"
                if container.exists():
                    listings[name] = subprocess.run([str(Path(sdk) / "dxc.exe"), "-dumpbin", str(container)],
                                                    capture_output=True, text=True).stdout.splitlines()
            if len(listings) == 2:
                diff = list(difflib.unified_diff(listings["reference"], listings["native"], "reference", "native", lineterm="", n=1))
                print("\n".join(diff[:160]))
        if result.returncode == 77:
            return 77
        assert result.returncode == 0, f"probe exit {result.returncode}"
        assert "native_heap_handle_probe: PASS" in result.stdout
        return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
