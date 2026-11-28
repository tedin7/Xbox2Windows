"""Public slot 62 and indirect paths with the opt-in builder gate and a source-built module."""
from pathlib import Path
import os
import subprocess
import sys


def run(executable, module, mode):
    env = dict(os.environ, X2W_COMPUTE_BUILDER_GATE="1", X2W_D3D12_PERMISSIVE="1",
               X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1")
    result = subprocess.run([str(Path(executable).resolve()), str(Path(module).resolve()), mode],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, (mode, result.returncode, result.stdout, result.stderr)
    return result.stderr


def main(executable, module):
    builder = run(executable, module, "builder")
    assert "compute builder gate" in builder, builder
    assert "Dispatch refused" in builder, builder
    assert "ExecuteIndirect refused" in builder, builder

    nonbuilder = run(executable, module, "nonbuilder")
    assert "compute builder gate" not in nonbuilder, nonbuilder

    completed = run(executable, module, "completed")
    assert "all have a desktop PSO" in completed, completed
    assert "requires a desktop compute pipeline and root" not in completed, completed
    assert "ExecuteIndirect refused" not in completed, completed
    print("builder gate: incomplete set refused; completed set dispatch reaches the list; non-builder untouched")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
