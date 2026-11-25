"""Restart the source-built provider client with the same storage root."""
import os
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory() as directory:
    env = dict(os.environ, X2W_STORAGE_PATH=".")
    writer = sys.argv[2] if len(sys.argv) > 2 else "write"
    for operation in (writer, "read"):
        result = subprocess.run([sys.argv[1], operation], cwd=directory, env=env)
        if result.returncode:
            sys.exit(result.returncode)
