"""Assemble a two-exit mesh container, run both CLI lowerers, and check the GPU draw."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from compute_store_check import find_llvm15, rejects
from mesh_dynamic_aot_draw_check import compile_shader, fixture
from mesh_dynamic_geometry_check import CODE, MESH_LIMITS, SECOND, SOURCE, WAVE64, decode
import mesh_dynamic_aot
import mesh_entry_aot
from pixel_aot import MESH_WAVE64, _read_named_container, container_code, disassemble
from pixel_aot_check import envelope
from resource_inventory import parts
from shader_profile_audit import classify


# After the shared barrier, wave 1 reaches its own S_ENDPGM. Wave 0 skips
# that exit and exports the vertices, attributes and primitive indices.
EXIT_BLOCK = """s_barrier
s_cmp_eq_u32 s8, 0
s_cbranch_scc0 retiring_wave
s_branch mesh_outputs
retiring_wave:
s_endpgm
mesh_outputs:"""


def mesh_container(code, *, stage=13):
    padded = code + bytes.fromhex("00009fbf") * (4 - (len(code) // 4 + 10) % 4)
    record = envelope(padded, directory_end=(40 + len(padded), 288 + len(padded)),
                      compute_width=128, mesh_limits=MESH_LIMITS, shader_stage=stage)
    return _read_named_container(record, len(b"fixture") + 2)


def verify(directory, kind, tool, objcopy, library):
    source, obj, raw = (directory / name for name in ("mesh.s", "mesh.o", "mesh.bin"))
    program = SOURCE.read_text(encoding="ascii")
    assert program.count("s_barrier\n") == 1
    source.write_text(program.replace("s_barrier\n", EXIT_BLOCK + "\n"), encoding="ascii")
    options = (["-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-mattr=" + WAVE64, "-filetype=obj"]
               if kind == "mc" else ["-target", "amdgcn-amd-amdpal", "-mcpu=gfx1013", "-mwavefrontsize64", "-c"])
    subprocess.run([tool, *options, source, "-o", obj], check=True)
    subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True)
    code = raw.read_bytes()
    decoded = disassemble(code, library, offsets=True, features=MESH_WAVE64)
    assert decoded == decode(code)
    exits = [pc for pc, op in decoded if op == "s_endpgm"]
    assert len(exits) == 2 and exits[-1] == len(code) - 4
    geometry = [0x2004, SECOND]
    container = directory / "mesh.dxbc"
    blob = mesh_container(code)
    container.write_bytes(blob)
    single = mesh_container(CODE)
    assert classify(blob, parts(blob), 13, library, None) == classify(single, parts(single), 13, library, None)
    words = directory / "words.json"
    words.write_text(json.dumps(geometry), encoding="ascii")
    outputs = []
    for tool_module, options, expected in (
        (mesh_entry_aot, ["--geometry-words", words], mesh_entry_aot.lower(decode(CODE), geometry, MESH_LIMITS)),
        (mesh_dynamic_aot, ["--geometry-word-count", "2"], mesh_dynamic_aot.lower(decode(CODE), 2, MESH_LIMITS)),
    ):
        output = directory / (tool_module.__name__ + ".hlsl")
        subprocess.run([sys.executable, "-B", tool_module.__file__, container, "--llvm", library,
                        "--output", output, *options], check=True)
        text = output.read_text(encoding="ascii")
        assert text == expected, tool_module.__name__
        outputs.append(text)

    def extract(binary, llvm=library, features=MESH_WAVE64):
        return container_code(parts(mesh_container(binary))[b"DXIL"], stage=13, llvm=llvm, features=features)

    def replace(pc, word):
        return code[:pc] + word + code[pc + 4:]

    nop = bytes.fromhex("000080bf")
    # Replacing the jump over the first exit makes the second exit unreachable.
    dead_path = mesh_container(replace(exits[0] - 4, nop))
    rejects("Unreachable code", mesh_entry_aot.decode_mesh, dead_path, library)
    rejects("Unverified multi-exit program", extract, code + code[-4:])
    allocation, = [pc for pc, op in decoded if op == "s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)"]
    rejects("missing mesh allocation", extract, replace(allocation, nop))
    # Keep the real allocation and replace the later, reachable barrier with
    # s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP). Only the allocation may fall through.
    barrier, = [pc for pc, op in decoded if op == "s_barrier"]
    rejects("Unverified multi-exit control flow: s_sendmsg", extract, replace(barrier, bytes.fromhex("030090bf")))
    rejects("Unverified multi-exit program", extract, code, llvm=None)
    rejects("Unverified multi-exit program", extract, code, features=b"+wavefrontsize32")
    compute = parts(mesh_container(code, stage=5))[b"DXIL"]
    rejects("Unverified multi-exit control flow: s_sendmsg", container_code, compute,
            stage=5, llvm=library, features=MESH_WAVE64)
    print(f"{len(code)} bytes: both CLIs and the profile audit match the single-exit mesh; refusal checks passed")
    return outputs


def draw(directory, dxc, renderer, adapter, outputs):
    pixel = compile_shader(dxc, directory, "solid", "float4 main() : SV_Target { return float4(1,0,1,1); }", "ps_6_0")
    reference = directory / "draw.bin"
    for name, source in zip(("static", "runtime"), outputs, strict=True):
        mesh = compile_shader(dxc, directory, name, source, "ms_6_5")
        command = [renderer, adapter, "--mesh-indexed", mesh, pixel, reference]
        triangle = ((0, 1, 2),)
        quad = (*triangle, (2, 1, 3))
        for indices, mismatches in ((quad, 0), (triangle, 528)):
            reference.write_bytes(fixture(0x2004, SECOND, indices))
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            if result.returncode == 77:
                print(result.stdout + result.stderr)
                return 77
            count = f"pixels=1024 mismatches={mismatches}"
            if (result.returncode != int(bool(mismatches)) or count not in result.stdout.splitlines()
                    or (mismatches and "Draw reference mismatch" not in result.stderr)):
                raise AssertionError(result.stdout + result.stderr)
            print(f"{adapter} {name}: 1024 pixels, {mismatches} expected mismatches")
    return 0


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        if len(sys.argv) == 5 and sys.argv[1] == "binary":
            verify(directory, "mc", *sys.argv[2:])
        elif len(sys.argv) == 5 and sys.argv[1] == "draw":
            llvm = find_llvm15()
            if llvm is None:
                print("skip: two-exit mesh draw requires LLVM 15 with AMDGPU")
                sys.exit(77)
            sys.exit(draw(directory, *sys.argv[2:], verify(directory, *llvm)))
        else:
            sys.exit("expected binary <llvm-mc> <llvm-objcopy> <libLLVM> or draw <dxc> <renderer> <adapter>")
