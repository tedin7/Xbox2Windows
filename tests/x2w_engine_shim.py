"""Makes every `import gfx2hlsl` that runs after this one the opt-in engine (tools/startup/gfx2hlsl_engine.py).

The engine tests are written like the default tests (they call G.ComputePipeline, G.MeshPipeline ...); importing this
module first makes G the engine module, so the same code drives the engine and the default translator is untouched.
"""
from pathlib import Path
import sys

_TOOLS = str(Path(__file__).resolve().parents[1] / 'tools' / 'startup')
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
import gfx2hlsl_engine  # noqa: E402

sys.modules['gfx2hlsl'] = gfx2hlsl_engine
