"""Verify that explicit hardware adapter filters cannot fall back to the default."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


with tempfile.TemporaryDirectory() as temp:
    work = Path(temp)
    output = work / 'readback.bin'
    for wanted, diagnostic in (
        ('no adapter has this name', 'No D3D12 hardware adapter matches XBOX2WINDOWS_ADAPTER='),
        ('x' * 128, 'XBOX2WINDOWS_ADAPTER is longer than 127 characters'),
    ):
        env = dict(os.environ, XBOX2WINDOWS_ADAPTER=wanted)
        # The runner must reject the filter before touching any shader inputs.
        run = subprocess.run([sys.argv[1], work / 'sdk', work / 'request', work / 'cache',
                              '0', output, '--direct'], env=env, capture_output=True, text=True, timeout=10)
        assert run.returncode == 2 and diagnostic in run.stderr, (run.returncode, run.stdout, run.stderr)
        assert not run.stdout and not output.exists(), run.stdout
print('Unmatched and overlong adapter filters were rejected before replay')
