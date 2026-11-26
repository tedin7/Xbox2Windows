"""The opt-in engine hook: default off, keys apart, one entry point.

tools/startup/gfx2hlsl.py hands over to tools/startup/gfx2hlsl_engine.py only when X2W_ENGINE=1. With the switch off the
default translator must not import the engine and must keep its keys; with it on the engine's keys carry the suffix that
src/startup_stubs/d3d12_draw.inl hashes the same way ("-engine1", after the revision string).
"""
import os
import re
from pathlib import Path
import sys
import types

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "startup"
sys.path.insert(0, str(TOOLS))

import gfx2hlsl as default  # noqa: E402

assert "gfx2hlsl_engine" not in sys.modules, "the default translator imported the engine on its own"
os.environ.pop("X2W_ENGINE", None)

# Switch on: main() is the engine's main and nothing else runs first.
fake = types.ModuleType("gfx2hlsl_engine")
fake.main = lambda: "engine main"
sys.modules["gfx2hlsl_engine"] = fake
os.environ["X2W_ENGINE"] = "1"
assert default.main() == "engine main", "X2W_ENGINE=1 did not hand over to the engine"
# Any other value keeps the default driver (it would fail on missing arguments, never reach the fake).
os.environ["X2W_ENGINE"] = "0"
try:
    default.main()
except (IndexError, SystemExit):
    pass
else:
    raise AssertionError("X2W_ENGINE=0 must keep the default driver")
del sys.modules["gfx2hlsl_engine"]
os.environ.pop("X2W_ENGINE", None)

import gfx2hlsl_engine as engine  # noqa: E402

# Keys: the same inputs give different keys, and the engine's follow the documented seed.
blobs = (b"vs", b"ps", b"root")
assert default.pipeline_key(*blobs) != engine.pipeline_key(*blobs)
assert default.compute_key(b"cs", b"data", b"root", b"desc") != engine.compute_key(b"cs", b"data", b"root", b"desc")
suffix = engine.ENGINE_KEY_SUFFIX
assert suffix == b"-engine1", suffix
# The runtime seeds the key with its graphics revision, then the suffix: the engine must use the same revision.
runtime = (TOOLS.parents[1] / "src" / "startup_stubs" / "d3d12_draw.inl").read_text(encoding="utf-8")
stub_revision = re.search(r'constexpr char revision\[\] = "([^"]+)";', runtime).group(1).encode()
revision = engine.fnv(suffix, engine.fnv(stub_revision))
assert engine.pipeline_key(*blobs) == engine.fnv(b"root", engine.fnv(b"ps", engine.fnv(b"vs", revision)))
assert engine.compute_key(b"cs", b"data", b"root", b"desc").bit_length() > 0
print("startup_engine_hook: PASS (default imports no engine, X2W_ENGINE=1 hands over, keys differ and follow the seed)")
