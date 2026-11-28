"""Reject a sampler word copied from EXEC after a 64-bit implicit write."""
from pathlib import Path
import subprocess
import sys
import tempfile

from native_direct_container_check import comparison_program, compute, words


def program(exec_op=None):
    code = comparison_program(dim=5)
    sampler_address = code[4]
    code[3:5] = [
        0x7e2c0500,  # v_readfirstlane_b32 s22, v0
        0xbe970380,  # s_mov_b32 s23, 0
        0xbefe04ff, sampler_address,  # s_mov_b64 exec, comparison sampler word
        *(() if exec_op is None else (exec_op,)),
        0xbe84037e,  # s_mov_b32 s4, exec_lo
    ]
    return compute(words(*code, 0xbf810000), threads=64, table=True, wave32=False)


def controls(cli, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        for name, exec_op in (
            ("constant-exec", None),
            ("saveexec", 0xbe982416),  # s_and_saveexec_b64 s[24:25], s[22:23]
            ("wrexec", 0xbe983a16),   # s_andn2_wrexec_b64 s[24:25], s[22:23]
        ):
            source = work / (name + ".cdx")
            output = source.with_suffix(".cso")
            source.write_bytes(program(exec_op))
            result = subprocess.run(
                [cli, sdk, work / "cache", "gfx1013-container-cs", source, output],
                capture_output=True, text=True,
            )
            if exec_op is None:
                assert result.returncode == 0 and output.read_bytes().startswith(b"DXBC"), (name, result.stderr)
            else:
                assert result.returncode != 0 and "nonconstant sampler" in result.stderr, (name, result.stderr)
                assert not output.exists(), output


if __name__ == "__main__":
    mode, cli, sdk = sys.argv[1:]
    {"controls": controls}[mode](cli, sdk)
