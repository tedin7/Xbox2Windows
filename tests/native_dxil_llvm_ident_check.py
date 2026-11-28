"""Verify direct DXIL !llvm.ident emission and D3D12 draw execution (#2915)."""
import ctypes
from ctypes import POINTER, Structure, byref, c_char_p, c_int32, c_size_t, c_uint32, c_void_p
from pathlib import Path
import re
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import uuid

from native_direct_container_check import (
    compute,
    interpolated_pixel,
    patched_pixel,
    publish,
    signature,
    signature_rows,
    vertex,
    words,
)


class _GUID(Structure):
    _fields_ = [
        ("Data1", c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def parse(cls, text: str) -> "_GUID":
        return cls.from_buffer_copy(uuid.UUID(text).bytes_le)


_CLSID_DXC_UTILS = _GUID.parse("6245d6af-66e0-48fd-80b4-4d271796748c")
_IID_IDXC_UTILS = _GUID.parse("4605c4cb-2019-492a-ada4-65f20bb7d67f")
_CLSID_DXC_ASSEMBLER = _GUID.parse("d728db68-f903-4f80-94cd-dccf76ec7151")
_IID_IDXC_ASSEMBLER = _GUID.parse("091f7a26-1c1f-4948-904b-e6e3a8a771d5")
_CLSID_DXC_VALIDATOR = _GUID.parse("8ca3e215-f728-4cf3-8cdd-88af917587a1")
_IID_IDXC_VALIDATOR = _GUID.parse("a6e82bd2-1fd7-4826-9811-2857e797f49a")


def _vfn(obj, index: int, restype, *argtypes):
    vtbl = ctypes.cast(ctypes.cast(obj, POINTER(c_void_p))[0], POINTER(c_void_p))
    return ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)(vtbl[index])


def _blob_bytes(blob: c_void_p) -> bytes:
    ptr = _vfn(blob, 3, c_void_p)(blob)
    size = _vfn(blob, 4, c_size_t)(blob)
    return ctypes.string_at(ptr, size)


def _take_result(op_result: c_void_p) -> tuple[bytes, c_void_p]:
    status = c_int32()
    assert _vfn(op_result, 3, c_int32, POINTER(c_int32))(op_result, byref(status)) == 0
    if status.value < 0:
        err = c_void_p()
        _vfn(op_result, 5, c_int32, POINTER(c_void_p))(op_result, byref(err))
        detail = _blob_bytes(err).decode("utf-8", "replace") if err.value else f"{status.value:#x}"
        raise RuntimeError(f"DXIL assembly/validation failed: {detail}")
    blob = c_void_p()
    assert _vfn(op_result, 4, c_int32, POINTER(c_void_p))(op_result, byref(blob)) == 0
    return _blob_bytes(blob), blob


def assemble_and_validate(sdk: Path, ir_text: str) -> bytes:
    dxc_dll = ctypes.WinDLL(str(sdk / "dxcompiler.dll"))
    dxil_dll = ctypes.WinDLL(str(sdk / "dxil.dll"))
    for dll in (dxc_dll, dxil_dll):
        dll.DxcCreateInstance.argtypes = [POINTER(_GUID), POINTER(_GUID), POINTER(c_void_p)]
        dll.DxcCreateInstance.restype = c_int32

    utils, assembler, validator = c_void_p(), c_void_p(), c_void_p()
    assert dxc_dll.DxcCreateInstance(byref(_CLSID_DXC_UTILS), byref(_IID_IDXC_UTILS), byref(utils)) == 0
    assert dxc_dll.DxcCreateInstance(byref(_CLSID_DXC_ASSEMBLER), byref(_IID_IDXC_ASSEMBLER), byref(assembler)) == 0
    assert dxil_dll.DxcCreateInstance(byref(_CLSID_DXC_VALIDATOR), byref(_IID_IDXC_VALIDATOR), byref(validator)) == 0

    encoded = ir_text.encode("utf-8")
    source_blob = c_void_p()
    assert _vfn(utils, 6, c_int32, c_char_p, c_uint32, c_uint32, POINTER(c_void_p))(
        utils, encoded, len(encoded), 65001, byref(source_blob)
    ) == 0
    assembled = c_void_p()
    assert _vfn(assembler, 3, c_int32, c_void_p, POINTER(c_void_p))(assembler, source_blob, byref(assembled)) == 0
    _, container_blob = _take_result(assembled)
    validated = c_void_p()
    assert _vfn(validator, 3, c_int32, c_void_p, c_uint32, POINTER(c_void_p))(
        validator, container_blob, 1, byref(validated)
    ) == 0
    cso_bytes, _ = _take_result(validated)
    return cso_bytes


def has_llvm_ident(disassembly: str, expected: str = "x2w") -> bool:
    match = re.search(r"^!llvm\.ident\s*=\s*!\{!(\d+)\}\s*$", disassembly, re.MULTILINE)
    if not match:
        return False
    node_id = match.group(1)
    node = re.search(rf"^!{node_id}\s*=\s*!\{{!\"([^\"]*)\"\}}\s*$", disassembly, re.MULTILINE)
    return bool(node and node.group(1) == expected)


def fullscreen_triangle_vs() -> bytes:
    # Fullscreen triangle (-1, -1), (4, -1), (-1, 4) from SV_VertexID in v0.
    program = [
        0x7E0C02F6,              # v_mov_b32_e32 v6, 4.0
        0x7D040081,              # v_cmp_eq_i32_e32 vcc, 1, v0
        0x02020CF3,              # v_cndmask_b32_e32 v1, -1.0, v6, vcc
        0x7D040082,              # v_cmp_eq_i32_e32 vcc, 2, v0
        0x02040CF3,              # v_cndmask_b32_e32 v2, -1.0, v6, vcc
        0x7E060280,              # v_mov_b32_e32 v3, 0
        0x7E0E02F2,              # v_mov_b32_e32 v7, 1.0
        0xF80008CF, 0x07030201,  # exp pos0 v1, v2, v3, v7 done
        0xBF810000,              # s_endpgm
    ]
    return vertex(program=program)


def check_containers(cli: Path, sdk: Path) -> None:
    dxc = sdk / "dxc.exe"
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        cache = work / "cache"

        def compile_and_dump(name: str, profile: str, payload: bytes) -> tuple[Path, str]:
            src = work / f"{name}.in"
            cso = work / f"{name}.cso"
            src.write_bytes(payload)
            run = subprocess.run([cli, sdk, cache, profile, src, cso], capture_output=True, text=True)
            assert run.returncode == 0, (name, profile, run.stdout, run.stderr)
            dump = subprocess.check_output([dxc, "-dumpbin", cso], text=True)
            assert has_llvm_ident(dump), f"{name} ({profile}) missing !llvm.ident:\n{dump}"
            return cso, dump

        cs_code = words(0x7E000280, 0xBF810000)
        compile_and_dump("compute", "gfx1013-container-cs", compute(cs_code))
        compile_and_dump("vertex", "gfx1013-container-vs", fullscreen_triangle_vs())
        _, ps_dump = compile_and_dump(
            "pixel",
            "gfx1013-container-ps",
            patched_pixel(1, [0x7E0002F2], input_signature=((b"OSG1", signature(b"SV_Target", 64)),)),
        )
        const_words = struct.pack("<8I", 0x7E0002F2, 0x7E0202F0, 0x7E0402FF, 0x3E800000, 0x7E0602F4, 0xF800180F, 0x03020100, 0xBF810000)
        _, const_ps_dump = compile_and_dump("direct_const_ps", "gfx1013-direct-constant-ps", const_words)

        # Rectangle geometry expansion emits VS, GS and PS through DxilModule.
        inputs_dir, rect_out = work / "rect_in", work / "rect_out"
        vs_rect = vertex(
            program=[0x7E020B00, 0xF80000CF, 0x01010101, 0xF8000A0F, 0x01010101, 0xBF810000],
            varying=True,
            extra=((b"OSG1", signature_rows([(b"TEXCOORD", 0, 3, 0, 15), (b"SV_Position", 1, 3, 1, 15)])),),
        )
        root = b"DXBC" + bytes(16) + words(1, 68, 1, 36) + b"RTS0" + words(24, 2, 0, 24, 0, 24, 0)
        stem = publish(inputs_dir, vs_rect, interpolated_pixel(), root)
        subprocess.run([cli, sdk, cache, "gfx1013-container-rect", "--requests", inputs_dir, rect_out], check=True)
        for stage in ("vs", "gs", "ps"):
            stage_dump = subprocess.check_output([dxc, "-dumpbin", rect_out / f"{stem}.{stage}.cso"], text=True)
            assert has_llvm_ident(stage_dump), f"rect.{stage} missing !llvm.ident"

        # Negative control: SDK validation accepts a module without !llvm.ident,
        # and has_llvm_ident rejects the stripped container across both direct emitters.
        for label, dump in (("container_ps", ps_dump), ("direct_const_ps", const_ps_dump)):
            ir_text = "\n".join(
                line for line in dump.splitlines() if not line.startswith(";") and not line.startswith("!llvm.ident")
            ) + "\n"
            stripped_cso = work / f"{label}_stripped.cso"
            stripped_cso.write_bytes(assemble_and_validate(sdk, ir_text))
            stripped_dump = subprocess.check_output([dxc, "-dumpbin", stripped_cso], text=True)
            assert not has_llvm_ident(stripped_dump), f"{label} negative control kept !llvm.ident"

    print("direct DXIL !llvm.ident container and negative controls passed (cs, vs, gs, ps, direct-constant-ps)")


def check_draws(cli: Path, raster: Path, sdk: Path, backend: str) -> None:
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        cache = work / "cache"
        vs_in, vs_cso = work / "vs.in", work / "vs.cso"
        vs_in.write_bytes(fullscreen_triangle_vs())
        subprocess.run([cli, sdk, cache, "gfx1013-container-vs", vs_in, vs_cso], check=True)

        # Case 1: gfx1013-direct-constant-ps -> (1.0, 0.5, 0.25, 2.0)
        const_in, const_cso = work / "const_ps.in", work / "const_ps.cso"
        const_words = [0x7E0002F2, 0x7E0202F0, 0x7E0402FF, 0x3E800000, 0x7E0602F4, 0xF800180F, 0x03020100, 0xBF810000]
        const_in.write_bytes(struct.pack("<8I", *const_words))
        subprocess.run([cli, sdk, cache, "gfx1013-direct-constant-ps", const_in, const_cso], check=True)

        expected = work / "expected.bin"
        expected.write_bytes(struct.pack("<4f", 1.0, 0.5, 0.25, 2.0) * 16)
        run = subprocess.run(
            [raster, backend, "--startup-pixel-words", vs_cso, const_cso, "8", "2", expected],
            capture_output=True,
            text=True,
        )
        if run.returncode == 77:
            sys.exit(77)
        assert run.returncode == 0, (run.stdout, run.stderr)
        if "debug_layer=enabled" in run.stdout:
            assert "debug_errors=0 debug_warnings=0" in run.stdout, run.stdout

        # Negative control 1: change alpha 2.0 -> 4.0 in gfx1013-direct-constant-ps; readback must fail.
        bad_words = list(const_words)
        bad_words[4] = 0x7E0602F6  # v3 = 4.0f
        bad_in, bad_cso = work / "bad_const_ps.in", work / "bad_const_ps.cso"
        bad_in.write_bytes(struct.pack("<8I", *bad_words))
        subprocess.run([cli, sdk, cache, "gfx1013-direct-constant-ps", bad_in, bad_cso], check=True)
        bad_run = subprocess.run(
            [raster, backend, "--startup-pixel-words", vs_cso, bad_cso, "8", "2", expected],
            capture_output=True,
            text=True,
        )
        assert bad_run.returncode != 0, "mutated gfx1013-direct-constant-ps unexpectedly matched"

        # Case 2: gfx1013-container-ps exporting distinct constants -> (0.5, 2.0, 1.0, 4.0)
        ps_in, ps_cso = work / "container_ps.in", work / "container_ps.cso"
        ps_seq = [0x7E0002F0, 0x7E0202F4, 0x7E0402F2, 0x7E0602F6]
        ps_in.write_bytes(
            patched_pixel(
                len(ps_seq),
                ps_seq,
                export=(0xF800180F, 0x03020100),
                input_signature=((b"OSG1", signature(b"SV_Target", 64)),),
            )
        )
        subprocess.run([cli, sdk, cache, "gfx1013-container-ps", ps_in, ps_cso], check=True)
        expected.write_bytes(struct.pack("<4f", 0.5, 2.0, 1.0, 4.0) * 16)
        run2 = subprocess.run(
            [raster, backend, "--startup-pixel-words", vs_cso, ps_cso, "8", "2", expected],
            capture_output=True,
            text=True,
        )
        assert run2.returncode == 0, (run2.stdout, run2.stderr)
        if "debug_layer=enabled" in run2.stdout:
            assert "debug_errors=0 debug_warnings=0" in run2.stdout, run2.stdout

        # Negative control 2: swap exported channels in gfx1013-container-ps; readback must fail.
        bad_ps_in, bad_ps_cso = work / "bad_container_ps.in", work / "bad_container_ps.cso"
        bad_ps_in.write_bytes(
            patched_pixel(
                len(ps_seq),
                ps_seq,
                export=(0xF800180F, 0x03020001),
                input_signature=((b"OSG1", signature(b"SV_Target", 64)),),
            )
        )
        subprocess.run([cli, sdk, cache, "gfx1013-container-ps", bad_ps_in, bad_ps_cso], check=True)
        bad_run2 = subprocess.run(
            [raster, backend, "--startup-pixel-words", vs_cso, bad_ps_cso, "8", "2", expected],
            capture_output=True,
            text=True,
        )
        assert bad_run2.returncode != 0, "mutated gfx1013-container-ps unexpectedly matched"

        print(run2.stdout.strip())


def main() -> None:
    mode = sys.argv[1]
    if mode == "container":
        _, _, cli, sdk = sys.argv
        check_containers(Path(cli), Path(sdk))
    elif mode == "draw":
        _, _, cli, raster, sdk, backend = sys.argv
        check_draws(Path(cli), Path(raster), Path(sdk), backend)
    else:
        raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
