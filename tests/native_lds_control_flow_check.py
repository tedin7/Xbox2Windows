"""Exercise the shared-memory frontend's reconvergence and rejection paths."""
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools/startup"))
from compile_lds_request import ComputeStage
from gfx2hlsl import Unsupported


barrier = "GroupMemoryBarrierWithGroupSync();"
blocks = [(0,[],("branch","(exec & 1) == 0",16)), (8,["v1 = 2;"],None),
          (16,[barrier],("end",))]
source = "\n".join(ComputeStage.body(SimpleNamespace(blocks=blocks)))
assert "!WaveActiveAnyTrue((exec&1u)!=0u)" in source
assert "}\n"+barrier+"\nif(pc==" in source

for invalid, expected in [
    ([(0,[],("branch","scc",16)),(8,[barrier],None),(16,[],("end",))],"conditional"),
    ([(0,[barrier],("jump",0))],"forward"),
]:
    try:
        ComputeStage.body(SimpleNamespace(blocks=invalid))
    except Unsupported as error:
        assert expected in str(error), error
    else:
        raise AssertionError("accepted invalid compute control flow")
print("uniform barrier and wave-level EXEC branch passed; divergent barrier and back edge rejected")

# A returning atomic replaces the entire destination, including a previous
# partial d16 result. Its address must still be readable when the registers alias.
def atomic_with_partial_register(address):
    stage = ComputeStage.__new__(ComputeStage)
    stage.pipe = SimpleNamespace(lds_words=128)
    stage.exec_known = True
    stage.used_v, stage.low16_only_v, stage.lines = set(), set(), []
    stage.pc = 0
    stage.code = struct.pack('<II', 0xd8800000, (7 << 24) | (6 << 8) | address)
    stage.set_v(7, '0u', low16_only=True)
    stage.lower('ds_add_rtn_u32', '', ['v7', f'v{address}', 'v6'], [])
    return stage.u('v7')


assert atomic_with_partial_register(1) == 'v7'
try:
    atomic_with_partial_register(7)
except Unsupported as error:
    assert 'whole-word read of v7' in str(error), error
else:
    raise AssertionError('accepted an undefined upper half in the atomic address')
