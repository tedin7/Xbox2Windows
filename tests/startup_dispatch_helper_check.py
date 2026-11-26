"""The shared dispatch() helper of the compute dispatch checks reports a hung consumer and a debug layer error.

usage: startup_dispatch_helper_check.py

A consumer that does not finish fails with its mode name and the stderr it printed, not with a bare ctest timeout; a
consumer whose debug layer printed an error (severity 0 or 1) fails in every mode, not only in the first run of a check.
"""
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest import mock

import startup_compute_dispatch_check as D


def fails_with(run, text, env):
    with tempfile.TemporaryDirectory() as temporary, mock.patch.object(subprocess, 'run', run):
        try:
            D.dispatch(Path(temporary), 'consumer', env, 'dispatch-demo')
        except AssertionError as error:
            assert text in str(error) and 'dispatch-demo' in str(error), str(error)
            return
    raise AssertionError('dispatch() accepted: ' + text)


def hung(*args, **kwargs):
    assert kwargs.get('timeout'), 'the consumer run has no timeout'
    raise subprocess.TimeoutExpired(args[0], kwargs['timeout'], stderr=b'last stage: CreateUnorderedAccessView')


def with_debug_error(*args, **kwargs):
    return SimpleNamespace(returncode=0, stdout='', stderr='x2w-d3d12-debug sev=1 id=555 view handle is not a descriptor\n')


fails_with(hung, 'last stage: CreateUnorderedAccessView', {})
fails_with(with_debug_error, 'view handle is not a descriptor', {'X2W_D3D12_DEBUG': '1'})
print('startup_dispatch_helper: a hung consumer and a debug layer error both fail the helper with the mode name')
