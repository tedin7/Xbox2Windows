"""Differential checks, alignment/header/MXCSR fault controls, dead-read ordering, mutation controls and rejection controls for XSAVE/XRSTOR AOT."""

import ctypes
from ctypes import wintypes
from pathlib import Path
import random
import struct
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Code, Decoder, Encoder, Instruction, MemoryOperand, Register as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_xsave_aot import lift_code, translate

GUARD_PRE = bytes(0xA0 + (i % 16) for i in range(64))
GUARD_POST = bytes(0xC0 + (i % 16) for i in range(64))
BUF_LEN = 960
DEFAULT_MXCSR = 0x1F80
FILL = b"\x5A" * BUF_LEN

ENTRY_ONLY_CANONICAL_EXPORTS = (
    "op_xsave_only",
    "op_xsave64_only",
)

CANONICAL_EXPORTS = ENTRY_ONLY_CANONICAL_EXPORTS + (
    "op_xrstor64_avx_xsaveopt64_avx",
)

XSAVEC_EXPORTS = (
    "op_xrstor64_xsavec",
    "op_xrstor64_xsavec64",
    "op_xrstor64_xsavec64_sse",
    "op_xrstor64_xsavec64_x87_avx",
)
DYNMASK_EXPORT = "op_xsavec64_dynmask"
XSAVEOPT_EXPORTS = (
    "op_xrstor64_xsaveopt",
    "op_xrstor64_xsaveopt64",
    "op_xrstor64_xsaveopt64_sse",
    "op_xrstor64_xsaveopt64_x87_avx",
    "op_xrstor64_xsaveopt64_avx",
)
AVX_XSAVE_EXPORT = "op_xrstor64_xsave64_avx"
MMX_READ_EXPORT = "op_xrstor64_movq_read_xsave64"
SSE_COMPACTED_CONTROL_EXPORT = "op_xrstor64_sse_compacted_avx_xsaveopt64_avx"
XSAVEOPT_DYNMASK_EXPORT = "op_xsaveopt64_dynmask"

# XSAVEC then XRSTOR of the same buffer: (xsavec RFBM, xrstor RFBM). The save mask leaves a component out of XCOMP_BV
# when it is not 7, and the restore mask selects which components come back from the compacted image.
COMPACTED_ROUNDTRIPS = {
    "op_xsavec_xrstor_all": (7, 7),
    "op_xsavec_xrstor_x87_avx": (5, 7),
    "op_xsavec_xrstor_x87_sse": (3, 7),
    "op_xsavec_xrstor_avx_only": (7, 4),
    "op_xsavec_xrstor_sse_only": (7, 2),
}
COMPACTED_DYNMASK_EXPORT = "op_xrstor64_compacted_dynmask"
COMPACTED_EXPORTS = tuple(COMPACTED_ROUNDTRIPS) + (COMPACTED_DYNMASK_EXPORT,)
ALTERED_COMPACTED_EXPORTS = ("op_xsavec_xrstor_all", "op_xsavec_xrstor_avx_only")
GP_MESSAGE = "access violation reading 0xffffffffffffffff"

DEFINED_EXPORTS = (
    "op_xrstor_xsave",
    "op_xrstor64_xsave64",
    "op_xrstor_xsave64",
    "op_xrstor64_xsave",
    "op_xrstor64_disp",
    "op_xrstor64_mask_sse",
    "op_vzeroupper_xsave64",
    "op_pipeline",
    AVX_XSAVE_EXPORT,
    MMX_READ_EXPORT,
) + XSAVEC_EXPORTS + XSAVEOPT_EXPORTS

EXPORTS = CANONICAL_EXPORTS + DEFINED_EXPORTS

ALTERED_DEFINED_EXPORTS = (
    "op_xrstor64_xsave64",
    "op_xrstor_xsave64",
    "op_xrstor64_disp",
    "op_pipeline",
    "op_xrstor64_xsavec64",
    "op_xrstor64_xsavec64_sse",
    "op_xrstor64_xsaveopt64",
    "op_xrstor64_xsaveopt64_sse",
)

ALTERED_CANONICAL_EXPORTS = ("op_xrstor64_mask_sse",)

# (rstor64, save64, save_rfbm, offset)
EXPORT_MODES = {
    "op_xsave_only": (True, False, 7, 0),
    "op_xsave64_only": (True, True, 7, 0),
    "op_xrstor_xsave": (False, False, 7, 0),
    "op_xrstor64_xsave64": (True, True, 7, 0),
    "op_xrstor_xsave64": (False, True, 7, 0),
    "op_xrstor64_xsave": (True, False, 7, 0),
    "op_xrstor64_disp": (True, True, 7, 64),
    "op_xrstor64_mask_sse": (True, True, 2, 0),
    "op_vzeroupper_xsave64": (True, True, 7, 0),
    "op_pipeline": (True, True, 7, 0),
    AVX_XSAVE_EXPORT: (True, True, 4, 0),
    MMX_READ_EXPORT: (True, True, 7, 0),
    "op_xrstor64_xsavec": (True, False, 7, 0),
    "op_xrstor64_xsavec64": (True, True, 7, 0),
    "op_xrstor64_xsavec64_sse": (True, True, 2, 0),
    "op_xrstor64_xsavec64_x87_avx": (True, True, 5, 0),
    "op_xrstor64_xsaveopt": (True, False, 7, 0),
    "op_xrstor64_xsaveopt64": (True, True, 7, 0),
    "op_xrstor64_xsaveopt64_sse": (True, True, 2, 0),
    "op_xrstor64_xsaveopt64_x87_avx": (True, True, 5, 0),
    "op_xrstor64_xsaveopt64_avx": (True, True, 4, 0),
    "op_xrstor64_avx_xsaveopt64_avx": (True, True, 4, 0),
    "op_xrstor64_sse_compacted_avx_xsaveopt64_avx": (True, True, 4, 0),
}


# --- Native XSAVE64/XRSTOR64 caller-state trampoline ---

def _encode_seq(*instructions) -> bytes:
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def clean_entry_bytes(mxcsr_mask: int, size: int) -> bytes:
    buf = bytearray(size)
    struct.pack_into("<HHBBH", buf, 0, 0x037F, 0, 0, 0, 0)
    struct.pack_into("<II", buf, 24, 0x1F80, mxcsr_mask)
    return bytes(buf)


class NativeHarness:
    """VirtualAlloc x64 caller harness preserving caller XMM6..15, YMM6..15, x87/MMX and MXCSR."""

    def __init__(self, area_size: int, mxcsr_mask: int):
        if area_size < 832:
            raise ValueError(f"Host XSAVE area is too small: {area_size}")
        # The trampoline saves every enabled host component. Guest encodings remain bounded to 832 bytes.
        self.area_size = area_size
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
        k32.VirtualAlloc.restype = ctypes.c_void_p
        k32.VirtualFree.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
        k32.VirtualFree.restype = ctypes.c_ubyte
        k32.RtlAddFunctionTable.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.c_uint64]
        k32.RtlAddFunctionTable.restype = ctypes.c_ubyte
        k32.RtlDeleteFunctionTable.argtypes = [ctypes.c_void_p]
        k32.RtlDeleteFunctionTable.restype = ctypes.c_ubyte
        self._k32 = k32

        # fn_save(rcx=buf): mov eax, -1; mov edx, -1; xsave64 [rcx]; ret
        save_code = _encode_seq(
            Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EAX, -1),
            Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EDX, -1),
            Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.RCX)),
            Instruction.create(Code.RETNQ),
        )
        # fn_rstor(rcx=buf): mov eax, -1; mov edx, -1; xrstor64 [rcx]; ret
        rstor_code = _encode_seq(
            Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EAX, -1),
            Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EDX, -1),
            Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
            Instruction.create(Code.RETNQ),
        )
        # 32-bit XSAVE/XRSTOR stubs for bounded RFBM=7 host FOP/FIP/FDP probing.
        save32_code = _encode_seq(
            Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EAX, 7),
            Instruction.create_reg_reg(Code.XOR_R32_RM32, R.EDX, R.EDX),
            Instruction.create_mem(Code.XSAVE_MEM, MemoryOperand(R.RCX)),
            Instruction.create(Code.RETNQ),
        )
        rstor32_code = _encode_seq(
            Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EAX, 7),
            Instruction.create_reg_reg(Code.XOR_R32_RM32, R.EDX, R.EDX),
            Instruction.create_mem(Code.XRSTOR_MEM, MemoryOperand(R.RCX)),
            Instruction.create(Code.RETNQ),
        )
        # fn_call(rcx=target_fn, rdx=args_block) with SEH unwind metadata:
        #   prolog: push rbx (1B); sub rsp, 32 (4B) -> 5B total
        call_code = _encode_seq(
            Instruction.create_reg(Code.PUSH_R64, R.RBX),
            Instruction.create_reg_i32(Code.SUB_RM64_IMM8, R.RSP, 32),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R10, R.RCX),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RBX, R.RDX),
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RCX, MemoryOperand(R.RBX, displ=24, displ_size=1)),
            Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EAX, -1),
            Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EDX, -1),
            Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.RCX)),
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RCX, MemoryOperand(R.RBX, displ=32, displ_size=1)),
            Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RCX, MemoryOperand(R.RBX, displ=0)),
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RDX, MemoryOperand(R.RBX, displ=8, displ_size=1)),
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.R8, MemoryOperand(R.RBX, displ=16, displ_size=1)),
            Instruction.create_reg(Code.CALL_RM64, R.R10),
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RCX, MemoryOperand(R.RBX, displ=24, displ_size=1)),
            Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EAX, -1),
            Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EDX, -1),
            Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
            Instruction.create_reg_i32(Code.ADD_RM64_IMM8, R.RSP, 32),
            Instruction.create_reg(Code.POP_R64, R.RBX),
            Instruction.create(Code.RETNQ),
        )
        rstor_off = len(save_code)
        save32_off = rstor_off + len(rstor_code)
        rstor32_off = save32_off + len(save32_code)
        call_off = rstor32_off + len(rstor32_code)
        blob = bytearray(544)
        blob[0:rstor_off] = save_code
        blob[rstor_off:save32_off] = rstor_code
        blob[save32_off:rstor32_off] = save32_code
        blob[rstor32_off:call_off] = rstor32_code
        blob[call_off : call_off + len(call_code)] = call_code
        # RUNTIME_FUNCTION at +512, UNWIND_INFO at +528 (push rbx + sub rsp, 32).
        struct.pack_into("<III", blob, 512, call_off, call_off + len(call_code), 528)
        struct.pack_into("<BBBBHH", blob, 528, 0x01, 5, 2, 0x00, 0x3205, 0x3001)

        mem = k32.VirtualAlloc(None, 4096, 0x3000, 0x40)
        if not mem:
            raise ctypes.WinError(ctypes.get_last_error())
        self._mem = mem
        ctypes.memmove(mem, bytes(blob), len(blob))
        self._rt_func = mem + 512
        if not k32.RtlAddFunctionTable(self._rt_func, 1, mem):
            raise ctypes.WinError(ctypes.get_last_error())
        self.save_fn = ctypes.CFUNCTYPE(None, ctypes.c_uint64)(mem)
        self.rstor_fn = ctypes.CFUNCTYPE(None, ctypes.c_uint64)(mem + rstor_off)
        self.save32_fn = ctypes.CFUNCTYPE(None, ctypes.c_uint64)(mem + save32_off)
        self.rstor32_fn = ctypes.CFUNCTYPE(None, ctypes.c_uint64)(mem + rstor32_off)
        self.call_fn = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_uint64)(mem + call_off)

        self._save_buf, self._save_ptr, _, _, _ = alloc_aligned_buffer(bytes(area_size), align=64)
        self._clean_buf, self._clean_ptr, _, _, _ = alloc_aligned_buffer(clean_entry_bytes(mxcsr_mask, area_size), align=64)
        self._args = (ctypes.c_uint64 * 5)()

    def close(self):
        if self._mem:
            self._k32.RtlDeleteFunctionTable(self._rt_func)
            self._k32.VirtualFree(self._mem, 0, 0x8000)
            self._mem = None

    def call_native(self, fn, ptr_a: int, ptr_b: int, ptr_out: int, entry_ptr: int | None = None) -> None:
        fn_addr = ctypes.cast(fn, ctypes.c_void_p).value
        self._args[0] = ptr_a
        self._args[1] = ptr_b
        self._args[2] = ptr_out
        self._args[3] = self._save_ptr
        self._args[4] = self._clean_ptr if entry_ptr is None else entry_ptr
        self.save_fn(self._save_ptr)
        try:
            self.call_fn(fn_addr, ctypes.addressof(self._args))
        finally:
            self.rstor_fn(self._save_ptr)

    def probe_xsave_err_ptrs(self, mxcsr_mask: int) -> dict[tuple[bool, bool, bool], tuple[bool, bool, bool, bool, bool]]:
        """Observe whether native XRSTOR* -> XSAVE* preserves FOP/FIP/FDP and selectors per (rstor64, save64, FSW.ES)."""
        caps: dict[tuple[bool, bool, bool], tuple[bool, bool, bool, bool, bool]] = {}
        seed_fop = 0x0357
        seed_fip = 0x00007FFF12345678
        seed_fdp = 0x00007FFF87654321
        exp_fop = struct.pack("<H", seed_fop)
        selector_seeds = (0x3B, 0x2B)
        for rstor64 in (False, True):
            eff_fip = seed_fip if rstor64 else (seed_fip & 0xFFFFFFFF)
            eff_fdp = seed_fdp if rstor64 else (seed_fdp & 0xFFFFFFFF)
            for save64 in (False, True):
                width = 8 if save64 else 4
                exp_fip = struct.pack("<Q", eff_fip)[:width]
                exp_fdp = struct.pack("<Q", eff_fdp)[:width]
                for es in (False, True):
                    img = make_xsave_image(
                        mxcsr_mask,
                        xstate_bv=7,
                        fsw=0x0080 if es else 0x0000,
                        fop=seed_fop,
                        fip=seed_fip,
                        fdp=seed_fdp,
                        fcs=0 if rstor64 else selector_seeds[0],
                        fds=0 if rstor64 else selector_seeds[1],
                        mxcsr=0x1F80,
                        seed=1,
                        poison_reserved=False,
                    )
                    in_buf, in_ptr, _, _, _ = alloc_aligned_buffer(img.ljust(self.area_size, b"\x00"))
                    out_buf, out_ptr, out_pre, out_off, out_post = alloc_aligned_buffer(bytes(self.area_size))
                    self.save_fn(self._save_ptr)
                    try:
                        (self.rstor_fn if rstor64 else self.rstor32_fn)(in_ptr)
                        (self.save_fn if save64 else self.save32_fn)(out_ptr)
                    finally:
                        self.rstor_fn(self._save_ptr)
                    raw = check_guards_and_read(out_buf, out_pre, out_off, out_post, "host FIP/FDP probe")
                    _ = in_buf
                    got_fop = raw[6:8]
                    got_fip = raw[8 : 8 + width]
                    got_fdp = raw[16 : 16 + width]
                    if got_fop not in (exp_fop, b"\x00\x00"):
                        raise AssertionError(
                            f"unexpected host FOP probe value for rstor64={rstor64}, save64={save64}, es={es}: {got_fop.hex()}"
                        )
                    if got_fip not in (exp_fip, b"\x00" * width):
                        raise AssertionError(
                            f"unexpected host FIP probe value for rstor64={rstor64}, save64={save64}, es={es}: {got_fip.hex()}"
                        )
                    if got_fdp not in (exp_fdp, b"\x00" * width):
                        raise AssertionError(
                            f"unexpected host FDP probe value for rstor64={rstor64}, save64={save64}, es={es}: {got_fdp.hex()}"
                        )
                    selectors = [False, False]
                    if not save64:
                        for index, offset in enumerate((12, 20)):
                            observed = struct.unpack_from("<H", raw, offset)[0]
                            expected = 0 if rstor64 else selector_seeds[index]
                            if observed not in (0, expected):
                                raise AssertionError(f"unexpected host selector at {offset}: {observed:#x}")
                            selectors[index] = not rstor64 and observed == expected
                    caps[(rstor64, save64, es)] = (
                        got_fop == exp_fop, got_fip == exp_fip, got_fdp == exp_fdp, *selectors
                    )
        return caps


# --- Independent Python XSAVE/XRSTOR reference model ---

class XStateModel:
    def __init__(self, mxcsr_mask: int, deprecates_fpu_cs_ds: bool):
        self.fcw = 0x037F
        self.fsw = 0
        self.ftw = 0
        self.fop = 0
        self.fip = 0
        self.fcs = 0
        self.fdp = 0
        self.fds = 0
        self.mxcsr = 0x1F80
        self.mxcsr_mask = mxcsr_mask
        self.deprecates_fpu_cs_ds = deprecates_fpu_cs_ds
        self.st = [bytearray(16) for _ in range(8)]
        self.xmm = [bytearray(16) for _ in range(16)]
        self.ymm_hi = [bytearray(16) for _ in range(16)]
        self.xinuse = 0

    def xrstor(
        self,
        buf: bytes,
        rfbm: int,
        is64: bool,
        *,
        check_subset: bool = True,
        check_reserved: bool = True,
        check_xcomp: bool = True,
        compacted_mxcsr: bool = True,
    ) -> None:
        """Standard and compacted XRSTOR; the keyword flags are the mutation controls, each off drops one rule.

        Compacted (XCOMP_BV[63] set, measured on hardware): the header must list only x87|SSE|AVX, XSTATE_BV must be
        a subset of it, and bytes [528..575] must be zero. A requested component restores when XSTATE_BV lists it
        and initializes otherwise (a component XCOMP_BV leaves out is initialized). MXCSR belongs to the SSE
        component: loaded when RFBM[1] and XSTATE_BV[1] are set, reset to 0x1F80 when only RFBM[1] is set, and left
        alone when RFBM[1] is clear. compacted_mxcsr=False applies the standard rule (loaded for RFBM[2:1] != 0).
        """
        rfbm &= 7
        xstate_bv, xcomp_bv, hdr_reserved = struct.unpack_from("<QQQ", buf, 512)
        compacted = bool(xcomp_bv >> 63)
        if compacted:
            listed = xcomp_bv & ~(1 << 63)
            if check_reserved and any(buf[528:576]):
                raise ValueError("compacted header bytes [528..575] are not zero")
            if check_xcomp and listed & ~7:
                raise ValueError(f"XCOMP_BV lists components outside x87|SSE|AVX: {listed:#x}")
            if check_subset and xstate_bv & ~listed:
                raise ValueError(f"XSTATE_BV {xstate_bv:#x} is not a subset of XCOMP_BV {listed:#x}")
            xstate_bv &= listed
            loads_mxcsr = bool(xstate_bv & 2) and bool(rfbm & 2)
            resets_mxcsr = bool(rfbm & 2) and not loads_mxcsr
            if not compacted_mxcsr:
                loads_mxcsr, resets_mxcsr = bool(rfbm & 6), False
        else:
            if xcomp_bv != 0 or hdr_reserved != 0 or (xstate_bv & ~7):
                raise ValueError("invalid XSAVE header")
            loads_mxcsr, resets_mxcsr = bool(rfbm & 6), False
        new_mxcsr = DEFAULT_MXCSR if resets_mxcsr else self.mxcsr
        if loads_mxcsr:
            new_mxcsr = struct.unpack_from("<I", buf, 24)[0]
            if new_mxcsr & ~self.mxcsr_mask:
                raise ValueError(f"reserved MXCSR bits set: {new_mxcsr:#x}")
        if rfbm & 1:
            if xstate_bv & 1:
                fcw_raw, fsw_raw, self.ftw, _, fop_raw = struct.unpack_from("<HHBBH", buf, 0)
                self.fcw = (fcw_raw & 0x1F7F) | 0x0040
                self.fsw = fsw_raw & 0x7F7F
                self.fop = fop_raw & 0x07FF
                if is64:
                    self.fip, self.fdp = struct.unpack_from("<QQ", buf, 8)
                    self.fcs = 0
                    self.fds = 0
                else:
                    fip32, fcs_raw = struct.unpack_from("<IH", buf, 8)
                    fdp32, fds_raw = struct.unpack_from("<IH", buf, 16)
                    self.fip = fip32
                    self.fdp = fdp32
                    if self.deprecates_fpu_cs_ds:
                        self.fcs = 0
                        self.fds = 0
                    else:
                        self.fcs = fcs_raw
                        self.fds = fds_raw
                top = (self.fsw >> 11) & 7
                for i in range(8):
                    phys = (top + i) & 7
                    self.st[phys][0:10] = buf[32 + i * 16 : 32 + i * 16 + 10]
                    self.st[phys][10:16] = b"\x00" * 6
                self.xinuse |= 1
            else:
                self.fcw = 0x037F
                self.fsw = 0
                self.ftw = 0
                self.fop = 0
                self.fip = 0
                self.fcs = 0
                self.fdp = 0
                self.fds = 0
                self.st = [bytearray(16) for _ in range(8)]
                self.xinuse &= ~1
        self.mxcsr = new_mxcsr
        if rfbm & 2:
            if xstate_bv & 2:
                for i in range(16):
                    self.xmm[i][:] = buf[160 + i * 16 : 160 + i * 16 + 16]
                self.xinuse |= 2
            else:
                self.xmm = [bytearray(16) for _ in range(16)]
                self.xinuse &= ~2
        if rfbm & 4:
            if xstate_bv & 4:
                for i in range(16):
                    self.ymm_hi[i][:] = buf[576 + i * 16 : 576 + i * 16 + 16]
                self.xinuse |= 4
            else:
                self.ymm_hi = [bytearray(16) for _ in range(16)]
                self.xinuse &= ~4

    def xsave(self, dst: bytearray, offset: int, rfbm: int, is64: bool, mxcsr_with_avx: bool = True) -> None:
        """XSAVE stores MXCSR and MXCSR_MASK when RFBM selects SSE or AVX, XSAVEC only with SSE."""
        rfbm &= 7
        if rfbm & 1:
            hdr = bytearray(24)
            struct.pack_into("<HHBBH", hdr, 0, self.fcw, self.fsw & 0x7F7F, self.ftw, 0, self.fop & 0x07FF)
            if is64:
                struct.pack_into("<QQ", hdr, 8, self.fip, self.fdp)
            else:
                fcs_out = 0 if self.deprecates_fpu_cs_ds else self.fcs
                fds_out = 0 if self.deprecates_fpu_cs_ds else self.fds
                struct.pack_into("<IHH", hdr, 8, self.fip & 0xFFFFFFFF, fcs_out, 0)
                struct.pack_into("<IHH", hdr, 16, self.fdp & 0xFFFFFFFF, fds_out, 0)
            dst[offset : offset + 24] = hdr
            st_area = bytearray(128)
            top = (self.fsw >> 11) & 7
            for i in range(8):
                phys = (top + i) & 7
                st_area[i * 16 : i * 16 + 10] = self.st[phys][0:10]
            dst[offset + 32 : offset + 160] = st_area
        if rfbm & (6 if mxcsr_with_avx else 2):
            struct.pack_into("<II", dst, offset + 24, self.mxcsr, self.mxcsr_mask)
        if rfbm & 2:
            for i in range(16):
                dst[offset + 160 + i * 16 : offset + 160 + i * 16 + 16] = self.xmm[i]
        old_bv = struct.unpack_from("<Q", dst, offset + 512)[0]
        new_bv = (old_bv & ~rfbm) | (self.xinuse & rfbm)
        struct.pack_into("<Q", dst, offset + 512, new_bv)
        if rfbm & 4:
            for i in range(16):
                dst[offset + 576 + i * 16 : offset + 576 + i * 16 + 16] = self.ymm_hi[i]

    def xsavec(self, dst: bytearray, offset: int, rfbm: int, is64: bool, track_mxcsr: bool = True) -> None:
        """SDM vol.1 13.4.3 compacted format: only XINUSE & RFBM components are stored.

        SDM vol.1 13.6: XINUSE[1] is also set while MXCSR differs from 0x1F80; track_mxcsr=False is the mutation
        control that forgets this rule.
        """
        rfbm &= 7
        sse_live = 2 if track_mxcsr and self.mxcsr != DEFAULT_MXCSR else 0
        used = rfbm & (self.xinuse | sse_live)
        self.xsave(dst, offset, used, is64, mxcsr_with_avx=False)
        struct.pack_into("<QQ", dst, offset + 512, used, (1 << 63) | rfbm)

    def xsaveopt(self, dst: bytearray, offset: int, rfbm: int, is64: bool, track_mxcsr: bool = True) -> None:
        """SDM vol.2 XSAVEOPT, init optimization only: components not in use are left untouched in memory.

        XSTATE_BV = (old & ~RFBM) | (XINUSE & RFBM); MXCSR and MXCSR_MASK are stored when RFBM selects SSE or AVX.
        track_mxcsr=False is the mutation control that forgets that a non-default MXCSR keeps SSE in use.
        """
        rfbm &= 7
        sse_live = 2 if track_mxcsr and self.mxcsr != DEFAULT_MXCSR else 0
        in_use = self.xinuse | sse_live
        self.xsave(dst, offset, rfbm & in_use, is64, mxcsr_with_avx=False)
        if rfbm & 6:
            struct.pack_into("<II", dst, offset + 24, self.mxcsr, self.mxcsr_mask)
        old_bv = struct.unpack_from("<Q", dst, offset + 512)[0]
        struct.pack_into("<Q", dst, offset + 512, (old_bv & ~rfbm) | (in_use & rfbm))

    def vzeroupper(self) -> None:
        self.ymm_hi = [bytearray(16) for _ in range(16)]
        self.xinuse &= ~4

    def enter_mmx(self) -> None:
        """Every MMX instruction but EMMS, one that only reads an MM register included, clears TOP and tags all valid."""
        self.ftw = 0xFF
        self.fsw &= ~0x3800
        self.xinuse |= 1

    def set_mm(self, idx: int, val: int) -> None:
        struct.pack_into("<QH", self.st[idx], 0, val & 0xFFFFFFFFFFFFFFFF, 0xFFFF)
        self.st[idx][10:16] = b"\x00" * 6
        self.enter_mmx()

    def get_mm(self, idx: int) -> int:
        self.enter_mmx()
        return struct.unpack_from("<Q", self.st[idx], 0)[0]


def ref_paddd(a: bytes, b: bytes) -> bytes:
    da = struct.unpack("<4I", a[:16])
    db = struct.unpack("<4I", b[:16])
    return struct.pack("<4I", *((x + y) & 0xFFFFFFFF for x, y in zip(da, db)))


def reference(
    name: str,
    a: bytes,
    b: bytes,
    mxcsr_mask: int,
    deprecates_cs_ds: bool,
    altered: bool = False,
) -> bytes:
    out = bytearray(FILL)
    m = XStateModel(mxcsr_mask, deprecates_cs_ds)
    if name == "op_xsave_only":
        m.xsave(out, 0, rfbm=7, is64=False)
    elif name == "op_xsave64_only":
        m.xsave(out, 0, rfbm=7, is64=True)
    elif name == "op_xrstor_xsave":
        m.xrstor(a, rfbm=7, is64=False)
        m.xsave(out, 0, rfbm=7, is64=False)
    elif name == "op_xrstor64_xsave64":
        m.xrstor(a, rfbm=7, is64=True)
        m.xsave(out, 0, rfbm=7, is64=not altered)
    elif name == "op_xrstor_xsave64":
        m.xrstor(a, rfbm=7, is64=altered)
        m.xsave(out, 0, rfbm=7, is64=True)
    elif name == "op_xrstor64_xsave":
        m.xrstor(a, rfbm=7, is64=True)
        m.xsave(out, 0, rfbm=7, is64=False)
    elif name == "op_xrstor64_disp":
        m.xrstor(a[64:], rfbm=7, is64=not altered)
        m.xsave(out, 64, rfbm=7, is64=True)
    elif name == "op_xrstor64_mask_sse":
        m.xrstor(a, rfbm=2, is64=True)
        m.xsave(out, 0, rfbm=7 if altered else 2, is64=True)
    elif name == "op_vzeroupper_xsave64":
        m.xrstor(a, rfbm=7, is64=True)
        m.vzeroupper()
        m.xsave(out, 0, rfbm=7, is64=True)
    elif name in XSAVEC_EXPORTS:
        m.xrstor(a, rfbm=7, is64=True)
        rfbm = {"op_xrstor64_xsavec64_sse": 2, "op_xrstor64_xsavec64_x87_avx": 5}.get(name, 7)
        if altered and name == "op_xrstor64_xsavec64":
            m.xsave(out, 0, rfbm=7, is64=True)
        else:
            m.xsavec(out, 0, rfbm=7 if altered else rfbm, is64=name != "op_xrstor64_xsavec")
    elif name == MMX_READ_EXPORT:
        m.xrstor(a, rfbm=7, is64=True)
        m.get_mm(3)
        m.xsave(out, 0, rfbm=7, is64=True)
    elif name == AVX_XSAVE_EXPORT:
        m.xrstor(a, rfbm=7, is64=True)
        m.xsave(out, 0, rfbm=4, is64=True)
    elif name == "op_xrstor64_avx_xsaveopt64_avx":
        m.xrstor(a, rfbm=4, is64=True)
        m.xsaveopt(out, 0, rfbm=4, is64=True)
    elif name == "op_xrstor64_sse_compacted_avx_xsaveopt64_avx":
        m.xrstor(a, rfbm=2, is64=True)
        m.xrstor(b, rfbm=4, is64=True)
        m.xsaveopt(out, 0, rfbm=4, is64=True)
    elif name in XSAVEOPT_EXPORTS:
        m.xrstor(a, rfbm=7, is64=True)
        rfbm = {"op_xrstor64_xsaveopt64_sse": 2, "op_xrstor64_xsaveopt64_x87_avx": 5, "op_xrstor64_xsaveopt64_avx": 4}.get(name, 7)
        if altered and name == "op_xrstor64_xsaveopt64":
            m.xsave(out, 0, rfbm=7, is64=True)
        else:
            m.xsaveopt(out, 0, rfbm=7 if altered else rfbm, is64=name != "op_xrstor64_xsaveopt")
    elif name == "op_pipeline":
        m.xrstor(a, rfbm=7, is64=True)
        mm0 = struct.unpack_from("<Q", b, 8 if altered else 0)[0]
        m.set_mm(0, mm0)
        rax = m.get_mm(0)
        m.set_mm(7, rax)
        ymm1_lo = ref_paddd(a[832:848], a[864:880])
        ymm1_hi = a[848:864]
        m.xmm[0][:] = a[832:848]
        m.ymm_hi[0][:] = a[848:864]
        m.xmm[1][:] = ymm1_lo
        m.ymm_hi[1][:] = ymm1_hi
        m.xinuse |= 2 | 4
        out[832:848] = ymm1_lo
        out[848:864] = ymm1_hi
        m.xsave(out, 0, rfbm=7, is64=True)
    else:
        raise ValueError(f"unknown export {name}")
    return bytes(out)


# --- Test vectors ---

def make_xsave_image(
    mxcsr_mask: int,
    *,
    xstate_bv: int = 7,
    fcw: int = 0x027F,
    fsw: int = 0,
    ftw: int = 0xAA,
    fop: int = 0x0357,
    fip: int = 0,
    fcs: int = 0,
    fdp: int = 0,
    fds: int = 0,
    mxcsr: int = 0x1F80,
    seed: int = 1,
    poison_reserved: bool = True,
) -> bytes:
    """Build a 960-byte XSAVE buffer; non-zero payloads when XSTATE_BV[i]==1 keep XINUSE CPU-independent."""
    buf = bytearray(b"\xAA" * BUF_LEN if poison_reserved else b"\x00" * BUF_LEN)
    # Valid standard-format header: XCOMP_BV = 0, header[16:64] = 0.
    buf[520:576] = b"\x00" * 56
    struct.pack_into("<Q", buf, 512, xstate_bv)
    struct.pack_into("<HHBBH", buf, 0, fcw, fsw, ftw, 0xBB if poison_reserved else 0, fop)
    struct.pack_into("<QQ", buf, 8, fip, fdp)
    if fcs or fds:
        struct.pack_into("<IHH", buf, 8, fip & 0xFFFFFFFF, fcs, 0)
        struct.pack_into("<IHH", buf, 16, fdp & 0xFFFFFFFF, fds, 0)
    struct.pack_into("<II", buf, 24, mxcsr & mxcsr_mask, 0xDEADBEEF if poison_reserved else mxcsr_mask)
    rng = random.Random(seed)
    for i in range(8):
        buf[32 + i * 16 : 32 + i * 16 + 10] = rng.randbytes(10)
        if poison_reserved:
            buf[32 + i * 16 + 10 : 32 + i * 16 + 16] = rng.randbytes(6)
    for i in range(16):
        buf[160 + i * 16 : 160 + i * 16 + 16] = rng.randbytes(16)
        buf[576 + i * 16 : 576 + i * 16 + 16] = rng.randbytes(16)
    # Ensure component payloads are distinctly non-zero when XSTATE_BV[i]==1.
    if (xstate_bv & 1) and buf[32:42] == b"\x00" * 10:
        buf[32] = 0x11
    if (xstate_bv & 2) and buf[160:176] == b"\x00" * 16:
        buf[160] = 0x22
    if (xstate_bv & 4) and buf[576:592] == b"\x00" * 16:
        buf[576] = 0x44
    buf[832:880] = rng.randbytes(48)
    return bytes(buf)


def xsave_pairs(mxcsr_mask: int):
    rng = random.Random(0x5A7E01)
    # Exercise all XSTATE_BV subsets {0, 1, 2, 3, 4, 6, 7} plus FIP/FDP/FCS/FDS/MXCSR edges.
    specs = [
        (0, 0x037F, 0x0000, 0x00, 0x0000, 0x0000000000000000, 0, 0x0000000000000000, 0, 0x1F80, 10),
        (1, 0x027F, 0x3800, 0xA5, 0xFFFF, 0x00007FFF12345678, 0, 0xFFFF800087654321, 0, 0x1F80, 11),
        (2, 0x037F, 0x0000, 0x00, 0x0000, 0x0000000000000000, 0, 0x0000000000000000, 0, 0x3F80, 12),
        (3, 0x0000, 0x8080, 0xFF, 0xF800, 0x00000000ABCDEF01, 0x1234, 0x0000000013572468, 0x5678, 0x5FBF, 13),
        (4, 0x037F, 0x0000, 0x00, 0x0000, 0x0000000000000000, 0, 0x0000000000000000, 0, 0x1F80, 14),
        (6, 0x037F, 0x0000, 0x00, 0x0000, 0x0000000000000000, 0, 0x0000000000000000, 0, 0x9FC0, 15),
        (7, 0xFFFF, 0x7F7F, 0x55, 0x1234, 0xFFFF800011223344, 0, 0x00007FFF55667788, 0, 0x5F80, 16),
        # XSTATE_BV[1] clear with a non-default MXCSR: SSE is still in use (SDM vol.1 13.6), so xsavec must save it.
        (0, 0x037F, 0x0000, 0x00, 0x0000, 0x0000000000000000, 0, 0x0000000000000000, 0, 0x3F80, 17),
        (4, 0x037F, 0x0000, 0x00, 0x0000, 0x0000000000000000, 0, 0x0000000000000000, 0, 0x3F80, 18),
        (1, 0x027F, 0x0000, 0xAA, 0x0357, 0x00007FFF12345678, 0, 0x00007FFF87654321, 0, 0x7FC0, 19),
    ]
    pairs = []
    for xbv, fcw, fsw, ftw, fop, fip, fcs, fdp, fds, mxcsr, seed in specs:
        a = make_xsave_image(
            mxcsr_mask, xstate_bv=xbv, fcw=fcw, fsw=fsw, ftw=ftw, fop=fop,
            fip=fip, fcs=fcs, fdp=fdp, fds=fds, mxcsr=mxcsr, seed=seed,
        )
        b = rng.randbytes(BUF_LEN)
        pairs.append((a, b))
    return pairs


def disp_pairs(mxcsr_mask: int):
    return [(b"\xEE" * 64 + a[: BUF_LEN - 64], b) for a, b in xsave_pairs(mxcsr_mask)]


# --- 64-byte guarded buffer helpers ---

def alloc_aligned_buffer(payload: bytes, align: int = 64, offset: int = 0):
    total = len(GUARD_PRE) + len(payload) + len(GUARD_POST) + align + offset
    buf = ctypes.create_string_buffer(total)
    base_addr = ctypes.addressof(buf)
    start_after_guard = base_addr + len(GUARD_PRE)
    pad = (align - start_after_guard % align) % align
    payload_off = len(GUARD_PRE) + pad + offset
    pre_off = payload_off - len(GUARD_PRE)
    post_off = payload_off + len(payload)
    buf[pre_off:payload_off] = GUARD_PRE
    buf[payload_off:post_off] = payload
    buf[post_off : post_off + len(GUARD_POST)] = GUARD_POST
    return buf, base_addr + payload_off, pre_off, payload_off, post_off


def check_guards_and_read(buf, pre_off: int, payload_off: int, post_off: int, label: str) -> bytes:
    raw = buf.raw
    if raw[pre_off:payload_off] != GUARD_PRE or raw[post_off : post_off + len(GUARD_POST)] != GUARD_POST:
        raise AssertionError(f"64-byte guard bytes corrupted around {label}")
    return raw[payload_off:post_off]


def invoke_native(harness: NativeHarness, fn, a: bytes, b: bytes, entry_ptr: int | None = None) -> bytes:
    buf_a, ptr_a, a_pre, a_off, a_post = alloc_aligned_buffer(a)
    buf_b, ptr_b, b_pre, b_off, b_post = alloc_aligned_buffer(b)
    buf_out, ptr_out, o_pre, o_off, o_post = alloc_aligned_buffer(FILL)
    harness.call_native(fn, ptr_a, ptr_b, ptr_out, entry_ptr=entry_ptr)
    check_guards_and_read(buf_a, a_pre, a_off, a_post, "native input_a")
    check_guards_and_read(buf_b, b_pre, b_off, b_post, "native input_b")
    return check_guards_and_read(buf_out, o_pre, o_off, o_post, "native output")


def expected_native_output(want: bytes, a: bytes, name: str, fp_caps: dict) -> bytes:
    """Adapt native pointer and selector expectations to the separate host probe."""
    rstor64, save64, save_rfbm, off = EXPORT_MODES[name]
    if (save_rfbm & 1) == 0:
        return want
    xstate_bv_in = struct.unpack_from("<Q", a, off + 512)[0]
    if (xstate_bv_in & 1) == 0:
        return want
    fsw_in = struct.unpack_from("<H", a, off + 2)[0]
    es = (fsw_in & 0x0080) != 0
    preserves_fop, preserves_fip, preserves_fdp, preserves_fcs, preserves_fds = fp_caps[(rstor64, save64, es)]
    width = 8 if save64 else 4
    out = bytearray(want)
    if not preserves_fop:
        out[off + 6 : off + 8] = b"\x00\x00"
    if not preserves_fip:
        out[off + 8 : off + 8 + width] = b"\x00" * width
    if not preserves_fdp:
        out[off + 16 : off + 16 + width] = b"\x00" * width
    if not save64:
        for offset, preserves in ((12, preserves_fcs), (20, preserves_fds)):
            start = off + offset
            out[start : start + 2] = a[start : start + 2] if preserves else bytes(2)
    return bytes(out)


def invoke_lifted(fn, a: bytes, b: bytes) -> bytes:
    buf_a, ptr_a, a_pre, a_off, a_post = alloc_aligned_buffer(a)
    buf_b, ptr_b, b_pre, b_off, b_post = alloc_aligned_buffer(b)
    buf_out, ptr_out, o_pre, o_off, o_post = alloc_aligned_buffer(FILL)
    fn(ptr_a, ptr_b, ptr_out)
    check_guards_and_read(buf_a, a_pre, a_off, a_post, "lifted input_a")
    check_guards_and_read(buf_b, b_pre, b_off, b_post, "lifted input_b")
    return check_guards_and_read(buf_out, o_pre, o_off, o_post, "lifted output")


def _xsave_header_flex(name: str) -> tuple[int, int] | None:
    """Return (header_offset, rfbm) for exports whose final XSAVE writes XSTATE_BV."""
    if name == "op_xrstor64_disp":
        return 64 + 512, 7
    if name == "op_xrstor64_mask_sse":
        return 512, 2
    if name in EXPORTS or name in COMPACTED_EXPORTS:
        return 512, 7
    return None


XSAVEOPT_COMPONENTS = (
    (1, ((0, 24), (32, 160))),
    (2, ((160, 416),)),
    (4, ((576, 832),)),
)
INIT_X87_HEADER = struct.pack("<HHBBH", 0x037F, 0, 0, 0, 0) + bytes(16)


def native_xsaveopt_matches(got: bytes, want: bytes, rfbm: int) -> bool:
    """Native xsaveopt comparison: components in use stay compared byte for byte.

    A component the reference leaves out may be stored by the host only in its initial configuration, with its
    XSTATE_BV bit set. SSE that is in use only through a non-default MXCSR holds zero XMM state, which a host
    without that rule may leave unwritten.
    """
    got_bv = struct.unpack_from("<Q", got, 512)[0]
    want_bv = struct.unpack_from("<Q", want, 512)[0]
    used = want_bv & rfbm
    loose = bytearray(want)
    for bit, spans in XSAVEOPT_COMPONENTS:
        if not rfbm & bit:
            continue
        stored = bool(got_bv & bit)
        if bit & used:
            mxcsr_only = (
                bit == 2
                and want[160:416] == bytes(256)
                and struct.unpack_from("<I", want, 24)[0] != DEFAULT_MXCSR
            )
            relaxed = mxcsr_only and not stored and all(got[lo:hi] == FILL[lo:hi] for lo, hi in spans)
        else:
            init = {(0, 24): INIT_X87_HEADER}
            relaxed = stored and all(got[lo:hi] == init.get((lo, hi), bytes(hi - lo)) for lo, hi in spans)
        if relaxed:
            for lo, hi in spans:
                loose[lo:hi] = got[lo:hi]
            loose[512:520] = struct.pack("<Q", (struct.unpack_from("<Q", loose, 512)[0] & ~bit) | (got_bv & bit))
    return got == bytes(loose)


def assert_equal_output(label: str, name: str, idx, got: bytes, want: bytes) -> None:
    if got == want:
        return
    # Per Intel SDM Vol. 1 Ch. 13.5.1, XSAVE may report XSTATE_BV[1:0] = 1 for initialized
    # legacy x87/SSE components when RFBM selects them (e.g. hybrid E-cores vs P-cores).
    if label == "native" and name in XSAVEC_EXPORTS:
        # The CPU may also store initialized x87/SSE slots, and MXCSR is only specified with the SSE slot.
        got_bv = struct.unpack_from("<Q", got, 512)[0]
        want_bv = struct.unpack_from("<Q", want, 512)[0]
        loose = bytearray(want)
        # Only components the reference leaves initialized may carry extra host bytes; active x87/SSE payloads
        # (control words, ST and XMM registers) stay compared. x87 pointer fields are adapted by expected_native_output.
        for bit, spans in ((1, ((0, 24), (32, 160))), (2, ((160, 416),))):
            if got_bv & bit and not want_bv & bit:
                for lo, hi in spans:
                    loose[lo:hi] = got[lo:hi]
        if not want_bv & 2:
            loose[24:32] = got[24:32]
        loose[512:520] = got[512:520]
        # A host-reported SSE component the reference leaves out must sit at its reset value: a non-default MXCSR
        # means the reference lost live state (SDM vol.1 13.6), so it is never tolerated.
        sse_extra_is_reset = not (got_bv & 2 and not want_bv & 2) or struct.unpack_from("<I", got, 24)[0] == DEFAULT_MXCSR
        if (
            sse_extra_is_reset
            and want_bv & ~got_bv == 0
            and got_bv & ~want_bv & ~3 == 0
            and got == bytes(loose)
        ):
            return
    elif label == "native" and (
        name in XSAVEOPT_EXPORTS or name in ("op_xrstor64_avx_xsaveopt64_avx", SSE_COMPACTED_CONTROL_EXPORT)
    ):
        if native_xsaveopt_matches(got, want, EXPORT_MODES[name][2]):
            return
    elif label == "native":
        flex_spec = _xsave_header_flex(name)
        if flex_spec is not None:
            hdr_off, rfbm = flex_spec
            got_bv = struct.unpack_from("<Q", got, hdr_off)[0]
            want_bv = struct.unpack_from("<Q", want, hdr_off)[0]
            flex_bits = (~want_bv) & 0x3 & rfbm
            if (got_bv & ~flex_bits) == want_bv and (
                got[:hdr_off] == want[:hdr_off] and got[hdr_off + 8 :] == want[hdr_off + 8 :]
            ):
                return
    diff = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)
    raise AssertionError(
        f"{label} {name} case {idx}: first mismatch at byte {diff}: "
        f"got {got[diff : diff + 16].hex()} != want {want[diff : diff + 16].hex()}"
    )


def native_exports_for_host(native: dict, has_xsavec: bool, has_xsaveopt: bool = True) -> dict:
    """Drop native xsavec/xsaveopt fixtures on hosts without them: executing them would raise #UD.

    The compacted XRSTOR fixtures go with XSAVEC: the compacted format is enumerated by the same CPUID bit, and
    without it a restore with XCOMP_BV[63] set always raises #GP. SSE_COMPACTED_CONTROL_EXPORT always restores a
    compacted image, so it is one of them.
    """
    return {
        name: fn
        for name, fn in native.items()
        if (
            has_xsavec
            or (name not in XSAVEC_EXPORTS and name not in COMPACTED_EXPORTS and name != SSE_COMPACTED_CONTROL_EXPORT)
        )
        and (
            has_xsaveopt
            or (name not in XSAVEOPT_EXPORTS and name not in ("op_xrstor64_avx_xsaveopt64_avx", SSE_COMPACTED_CONTROL_EXPORT))
        )
    }


def bind_exports(dll_path: Path, prefix: str = "", names=EXPORTS):
    lib = ctypes.CDLL(str(dll_path.resolve()))
    funcs = {}
    for name in names:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = None
        funcs[name] = fn
    return lib, funcs


# --- Runtime #GP / alignment / header / MXCSR / dead-read / host-state controls ---

def check_fault_controls(
    harness: NativeHarness, native_fns, lifted_fns, canonical_fns, mxcsr_mask: int
) -> tuple[int, int]:
    valid_a = make_xsave_image(mxcsr_mask, xstate_bv=7, fip=0x00007FFF11223344, fdp=0x00007FFF55667788, seed=99)

    misalign_checks = 0
    for off in (16, 8):
        for name in ENTRY_ONLY_CANONICAL_EXPORTS:
            for label, is_native, fn in (("native", True, native_fns[name]), ("canonical-lifted", False, canonical_fns[name])):
                buf_a, ptr_a, _, _, _ = alloc_aligned_buffer(valid_a)
                buf_out, ptr_out, pre, p_off, post = alloc_aligned_buffer(FILL, offset=off)
                try:
                    if is_native:
                        harness.call_native(fn, ptr_a, ptr_a, ptr_out)
                    else:
                        fn(ptr_a, ptr_a, ptr_out)
                except OSError as exc:
                    assert "access violation reading 0xffffffffffffffff" in str(exc).lower(), (name, label, off, exc)
                else:
                    raise AssertionError(f"{label} {name} accepted +{off} misaligned output")
                assert check_guards_and_read(buf_out, pre, p_off, post, label) == FILL
                _ = buf_a
                misalign_checks += 1

        for name in XSAVEC_EXPORTS + XSAVEOPT_EXPORTS:
            for label, is_native, fn in (("native", True, native_fns.get(name)), ("lifted", False, lifted_fns[name])):
                if fn is None:
                    continue  # host without XSAVEC/XSAVEOPT: native fixture is not executed
                buf_a, ptr_a, _, _, _ = alloc_aligned_buffer(valid_a)
                buf_out, ptr_out, pre, p_off, post = alloc_aligned_buffer(FILL, offset=off)
                try:
                    if is_native:
                        harness.call_native(fn, ptr_a, ptr_a, ptr_out)
                    else:
                        fn(ptr_a, ptr_a, ptr_out)
                except OSError as exc:
                    assert "access violation reading 0xffffffffffffffff" in str(exc).lower(), (name, label, off, exc)
                else:
                    raise AssertionError(f"{label} {name} accepted +{off} misaligned xsavec/xsaveopt output")
                assert check_guards_and_read(buf_out, pre, p_off, post, label) == FILL
                _ = buf_a
                misalign_checks += 1

        for name in ("op_xrstor_xsave", "op_xrstor64_xsave64"):
            for label, is_native, fn in (("native", True, native_fns[name]), ("lifted", False, lifted_fns[name])):
                buf_a, ptr_a, _, _, _ = alloc_aligned_buffer(valid_a, offset=off)
                buf_out, ptr_out, pre, p_off, post = alloc_aligned_buffer(FILL)
                try:
                    if is_native:
                        harness.call_native(fn, ptr_a, ptr_a, ptr_out)
                    else:
                        fn(ptr_a, ptr_a, ptr_out)
                except OSError as exc:
                    assert "access violation reading 0xffffffffffffffff" in str(exc).lower(), (name, label, off, exc)
                else:
                    raise AssertionError(f"{label} {name} accepted +{off} misaligned input")
                assert check_guards_and_read(buf_out, pre, p_off, post, label) == FILL
                _ = buf_a
                misalign_checks += 1

    bad_xcomp = bytearray(valid_a)
    struct.pack_into("<Q", bad_xcomp, 520, 1)  # nonzero XCOMP_BV in standard format
    bad_reserved = bytearray(valid_a)
    bad_reserved[528] = 1  # nonzero header[16:23]
    bad_xstate_bv = bytearray(valid_a)
    struct.pack_into("<Q", bad_xstate_bv, 512, 1 << 63)  # reserved XCR0 bit outside ~7
    bad_mxcsr = bytearray(valid_a)
    struct.pack_into("<I", bad_mxcsr, 24, 0x11F80)  # reserved MXCSR bit 16 set

    header_mxcsr_checks = 0
    for tag, bad_img, exports in (
        ("xcomp_bv", bytes(bad_xcomp), ("op_xrstor_xsave", "op_xrstor64_xsave64")),
        ("reserved_hdr", bytes(bad_reserved), ("op_xrstor_xsave", "op_xrstor64_xsave64")),
        ("unsupported_xstate_bv", bytes(bad_xstate_bv), ("op_xrstor_xsave", "op_xrstor64_xsave64")),
        ("invalid_mxcsr", bytes(bad_mxcsr), ("op_xrstor64_xsave64", "op_xrstor64_mask_sse")),
    ):
        for name in exports:
            for label, is_native, fn in (("native", True, native_fns[name]), ("lifted", False, lifted_fns[name])):
                buf_a, ptr_a, _, _, _ = alloc_aligned_buffer(bad_img)
                buf_out, ptr_out, pre, p_off, post = alloc_aligned_buffer(FILL)
                try:
                    if is_native:
                        harness.call_native(fn, ptr_a, ptr_a, ptr_out)
                    else:
                        fn(ptr_a, ptr_a, ptr_out)
                except OSError as exc:
                    assert "access violation reading 0xffffffffffffffff" in str(exc).lower(), (tag, name, label, exc)
                else:
                    raise AssertionError(f"{label} {name} accepted {tag} without #GP")
                assert check_guards_and_read(buf_out, pre, p_off, post, label) == FILL
                _ = buf_a
                header_mxcsr_checks += 1

    return misalign_checks, header_mxcsr_checks


def check_xsavec_dynamic_mask(
    fn, mxcsr_mask: int, deprecates_cs_ds: bool, export: str = DYNMASK_EXPORT, save: str = "xsavec"
) -> int:
    """Request masks arrive through RAX/RDX: valid ones match the model, unsupported bits fault with no store."""
    a = xsave_pairs(mxcsr_mask)[6][0]
    for mask in (1, 2, 3, 4, 6):
        want = bytearray(FILL)
        model = XStateModel(mxcsr_mask, deprecates_cs_ds)
        model.xrstor(a, rfbm=7, is64=True)
        getattr(model, save)(want, 0, rfbm=mask, is64=True)
        got = invoke_lifted(fn, a, struct.pack("<QQ", mask, 0).ljust(BUF_LEN, b"\x00"))
        assert_equal_output("lifted", export, mask, got, bytes(want))
    bad_masks = ((8, 0), (0x80, 0), (0xFFFFFFF8, 0), (7, 1), (0, 1 << 31))
    for lo, hi in bad_masks:
        buf_a, ptr_a, _, _, _ = alloc_aligned_buffer(a)
        buf_b, ptr_b, _, _, _ = alloc_aligned_buffer(struct.pack("<QQ", lo, hi).ljust(BUF_LEN, b"\x00"))
        buf_out, ptr_out, pre, p_off, post = alloc_aligned_buffer(FILL)
        try:
            fn(ptr_a, ptr_b, ptr_out)
        except OSError as exc:
            assert "access violation reading 0xffffffffffffffff" in str(exc).lower(), (lo, hi, exc)
        else:
            raise AssertionError(f"lifted {save} accepted unsupported request mask {hi:#x}:{lo:#x}")
        assert check_guards_and_read(buf_out, pre, p_off, post, "dynmask") == FILL
        _ = (buf_a, buf_b)
    return len(bad_masks)


def check_discarded_read(
    harness: NativeHarness,
    native_path: Path,
    lifted_path: Path,
    mxcsr_mask: int,
    deprecates_cs_ds: bool,
    fp_caps: dict,
) -> None:
    payload = make_xsave_image(mxcsr_mask, xstate_bv=7, fip=0x00007FFF12345678, fdp=0x00007FFF87654321, seed=77)
    bad_mxcsr = bytearray(payload)
    struct.pack_into("<I", bad_mxcsr, 24, 0x11F80)
    bad_xcomp = bytearray(payload)
    struct.pack_into("<Q", bad_xcomp, 520, 1)
    bad_reserved = bytearray(payload)
    bad_reserved[528] = 0x80

    _, n_fns = bind_exports(native_path, "", ("op_discarded_read",))
    _, l_fns = bind_exports(lifted_path, "lifted_", ("op_discarded_read",))

    want_first = reference("op_xrstor64_xsave64", payload, payload, mxcsr_mask, deprecates_cs_ds)
    expected_ok = bytearray(want_first)
    expected_ok[832:864] = payload[576:608]
    native_want_first = expected_native_output(want_first, payload, "op_xrstor64_xsave64", fp_caps)
    native_expected_ok = expected_native_output(bytes(expected_ok), payload, "op_xrstor64_xsave64", fp_caps)

    for label, is_native, fn in (("native", True, n_fns["op_discarded_read"]), ("lifted", False, l_fns["op_discarded_read"])):
        buf_a, ptr_a, _, _, _ = alloc_aligned_buffer(payload)
        buf_misalign, ptr_misalign, _, _, _ = alloc_aligned_buffer(payload, offset=16)
        buf_mxcsr, ptr_mxcsr, _, _, _ = alloc_aligned_buffer(bytes(bad_mxcsr))
        buf_xcomp, ptr_xcomp, _, _, _ = alloc_aligned_buffer(bytes(bad_xcomp))
        buf_res, ptr_res, _, _, _ = alloc_aligned_buffer(bytes(bad_reserved))

        # Readable valid rdx -> both [0..831] and [832..863] written.
        out_ok, ptr_ok, pre, off, post = alloc_aligned_buffer(FILL)
        if is_native:
            harness.call_native(fn, ptr_a, ptr_a, ptr_ok)
        else:
            fn(ptr_a, ptr_a, ptr_ok)
        assert_equal_output(
            label,
            "op_discarded_read",
            "readable",
            check_guards_and_read(out_ok, pre, off, post, label),
            native_expected_ok if is_native else bytes(expected_ok),
        )

        for mode, bad_ptr, is_gp in (
            ("unmapped", 0, False),
            ("misaligned", ptr_misalign, True),
            ("invalid_mxcsr", ptr_mxcsr, True),
            ("nonzero_xcomp_bv", ptr_xcomp, True),
            ("nonzero_reserved_hdr", ptr_res, True),
        ):
            out_err, ptr_err, pre, off, post = alloc_aligned_buffer(FILL)
            try:
                if is_native:
                    harness.call_native(fn, ptr_a, bad_ptr, ptr_err)
                else:
                    fn(ptr_a, bad_ptr, ptr_err)
            except OSError as exc:
                needle = "access violation reading 0xffffffffffffffff" if is_gp else "access violation reading"
                assert needle in str(exc).lower(), (label, mode, exc)
            else:
                raise AssertionError(f"{label} op_discarded_read discarded {mode} fault")
            assert_equal_output(
                label,
                "op_discarded_read",
                mode,
                check_guards_and_read(out_err, pre, off, post, label),
                native_want_first if is_native else want_first,
            )

        _ = buf_a, buf_misalign, buf_mxcsr, buf_xcomp, buf_res


def check_lifted_preserves_host_state(harness: NativeHarness, lifted_fns, mxcsr_mask: int) -> None:
    """Verify lifted functions do not touch caller host XMM6..15, YMM6..15, x87 or MXCSR."""
    host_img = make_xsave_image(mxcsr_mask, xstate_bv=7, mxcsr=0x3F80, seed=333)
    guest_img = make_xsave_image(mxcsr_mask, xstate_bv=7, mxcsr=0x5F80, seed=444)

    buf_host, ptr_host, _, _, _ = alloc_aligned_buffer(host_img.ljust(harness.area_size, b"\x00"))
    buf_guest, ptr_guest, _, _, _ = alloc_aligned_buffer(guest_img)
    buf_before, ptr_before, b_pre, b_off, b_post = alloc_aligned_buffer(bytes(harness.area_size))
    buf_after, ptr_after, a_pre, a_off, a_post = alloc_aligned_buffer(bytes(harness.area_size))
    buf_out, ptr_out, _, _, _ = alloc_aligned_buffer(FILL)

    harness.save_fn(harness._save_ptr)
    try:
        harness.rstor_fn(ptr_host)
        harness.save_fn(ptr_before)
        lifted_fns["op_xrstor64_xsave64"](ptr_guest, ptr_guest, ptr_out)
        harness.save_fn(ptr_after)
    finally:
        harness.rstor_fn(harness._save_ptr)

    before_bytes = check_guards_and_read(buf_before, b_pre, b_off, b_post, "host_before")[:832]
    after_bytes = check_guards_and_read(buf_after, a_pre, a_off, a_post, "host_after")[:832]
    # Check Windows x64 nonvolatile caller state: MXCSR, x87/MMX ST, and XMM6..15[127:0].
    for lo, hi, desc in (
        (24, 28, "MXCSR"),
        (32, 160, "x87/MMX ST"),
        (160 + 6 * 16, 416, "XMM6..15"),
    ):
        assert before_bytes[lo:hi] == after_bytes[lo:hi], f"lifted function mutated caller {desc}"
    _ = buf_host, buf_guest, buf_out


def check_entry_state_contracts(
    harness: NativeHarness,
    native_fns,
    native_altered_fns,
    lifted_lib,
    lifted_fns,
    altered_lib,
    canonical_lib,
    canonical_fns,
    canonical_altered_lib,
    canonical_altered_fns,
    compacted_lib,
    compacted_fns,
    compacted_altered_lib,
    mxcsr_mask: int,
    deprecates_cs_ds: bool,
    has_xsavec: bool,
) -> None:
    """Verify explicit entry-state contract metadata and non-default entry controls.

    has_xsavec: the host restores compacted images natively (CPUID.(0xD,1):EAX[1]); without it the native
    compacted restore would #GP, so only the standard-image native comparison runs.
    """
    for lib, expected_profile, expected_flag, expected_compacted in (
        (lifted_lib, b"defined", 0, 0),
        (altered_lib, b"defined", 0, 0),
        (canonical_lib, b"canonical", 1, 1),
        (canonical_altered_lib, b"canonical", 1, 0),
        (compacted_lib, b"defined", 0, 1),
        (compacted_altered_lib, b"defined", 0, 1),
    ):
        lib.lifted_entry_state_profile.restype = ctypes.c_char_p
        lib.lifted_requires_canonical_entry.restype = ctypes.c_uint32
        lib.lifted_allows_compacted_xrstor.restype = ctypes.c_uint32
        assert lib.lifted_entry_state_profile() == expected_profile
        assert lib.lifted_requires_canonical_entry() == expected_flag
        assert lib.lifted_allows_compacted_xrstor() == expected_compacted

    # 1) Non-default incoming SSE state on entry-only XSAVE/XSAVE64 exports:
    #    SSE in use (XSTATE_BV=2), MXCSR=0x3F80, low XMM0 lane=0x3F800000.
    nondefault_sse = bytearray(clean_entry_bytes(mxcsr_mask, harness.area_size))
    struct.pack_into("<Q", nondefault_sse, 512, 2)
    struct.pack_into("<I", nondefault_sse, 24, 0x3F80 & mxcsr_mask)
    struct.pack_into("<I", nondefault_sse, 160, 0x3F800000)
    sse_buf, sse_ptr, _, _, _ = alloc_aligned_buffer(bytes(nondefault_sse), align=64)
    dummy = b"\x00" * BUF_LEN
    for name in ENTRY_ONLY_CANONICAL_EXPORTS:
        got_n = invoke_native(harness, native_fns[name], dummy, dummy, entry_ptr=sse_ptr)
        got_l = invoke_lifted(canonical_fns[name], dummy, dummy)
        assert struct.unpack_from("<I", got_n, 24)[0] == (0x3F80 & mxcsr_mask)
        assert struct.unpack_from("<I", got_n, 160)[0] == 0x3F800000
        assert (struct.unpack_from("<Q", got_n, 512)[0] & 2) == 2
        assert struct.unpack_from("<I", got_l, 24)[0] == 0x1F80
        assert struct.unpack_from("<I", got_l, 160)[0] == 0
        assert (struct.unpack_from("<Q", got_l, 512)[0] & 7) == 0
        assert got_n != got_l
    _ = sse_buf

    # 2) Non-default incoming x87+AVX state on partially initialized restore/save:
    #    x87 in use (FCW=0x027F, ST0!=0) and AVX in use (YMM0_hi!=0), XSTATE_BV=1|4.
    nondefault_partial = bytearray(clean_entry_bytes(mxcsr_mask, harness.area_size))
    struct.pack_into("<H", nondefault_partial, 0, 0x027F)
    nondefault_partial[32] = 0x11
    struct.pack_into("<I", nondefault_partial, 576, 0x11223344)
    struct.pack_into("<Q", nondefault_partial, 512, 1 | 4)
    part_buf, part_ptr, _, _, _ = alloc_aligned_buffer(bytes(nondefault_partial), align=64)

    sse_in = make_xsave_image(mxcsr_mask, xstate_bv=2, mxcsr=0x5F80 & mxcsr_mask, seed=202)
    # Defined profile (xrstor(2) -> xsave(2)) never observes incoming x87/AVX:
    want_def = reference("op_xrstor64_mask_sse", sse_in, dummy, mxcsr_mask, deprecates_cs_ds)
    got_n_def = invoke_native(harness, native_fns["op_xrstor64_mask_sse"], sse_in, dummy, entry_ptr=part_ptr)
    got_l_def = invoke_lifted(lifted_fns["op_xrstor64_mask_sse"], sse_in, dummy)
    assert_equal_output("native_nondefault_entry", "op_xrstor64_mask_sse", 0, got_n_def, want_def)
    assert_equal_output("lifted_nondefault_entry", "op_xrstor64_mask_sse", 0, got_l_def, want_def)

    # Altered partial path (xrstor(2) -> xsave(7)) observes incoming x87/AVX natively,
    # while canonical-lifted documents and supplies canonical entry values:
    got_n_alt = invoke_native(
        harness, native_altered_fns["op_xrstor64_mask_sse"], sse_in, dummy, entry_ptr=part_ptr
    )
    got_l_alt = invoke_lifted(canonical_altered_fns["op_xrstor64_mask_sse"], sse_in, dummy)
    assert struct.unpack_from("<H", got_n_alt, 0)[0] == 0x027F
    assert got_n_alt[32] == 0x11
    assert struct.unpack_from("<I", got_n_alt, 576)[0] == 0x11223344
    assert (struct.unpack_from("<Q", got_n_alt, 512)[0] & 5) == 5
    assert struct.unpack_from("<H", got_l_alt, 0)[0] == 0x037F
    assert got_l_alt[32] == 0
    assert struct.unpack_from("<I", got_l_alt, 576)[0] == 0
    assert (struct.unpack_from("<Q", got_l_alt, 512)[0] & 5) == 0
    assert got_n_alt != got_l_alt

    # 3) Non-default entry-MXCSR control for AVX-only XRSTOR(4) -> XSAVEOPT64(4):
    #    - Default defined standard-only path accepts AVX-only XRSTOR(4) -> XSAVEOPT64(4), loads 0x5F80 from
    #      standard_avx (ignoring caller's 0x3F80), and #GPs on compacted_avx.
    #    - Opt-in compacted + canonical path handles both: compacted_avx leaves caller's 0x3F80 untouched
    #      natively while canonical-lifted supplies 0x1F80; standard_avx loads 0x5F80.
    #    - Opt-in compacted + defined SSE-defined positive (XRSTOR(2) -> XRSTOR(4) -> XSAVEOPT64(4))
    #      loads 0x5F80 from sse_in before compacted_avx and matches native.
    compacted_avx = compacted_image(mxcsr_mask, 4, COMPACTED_BIT | 4, 4, mxcsr=0x5F80 & mxcsr_mask, seed=303)
    standard_avx = make_xsave_image(mxcsr_mask, xstate_bv=4, mxcsr=0x5F80 & mxcsr_mask, seed=304)
    got_l_def_std = invoke_lifted(lifted_fns["op_xrstor64_avx_xsaveopt64_avx"], standard_avx, dummy)
    assert struct.unpack_from("<I", got_l_def_std, 24)[0] == (0x5F80 & mxcsr_mask)
    def_comp_faulted, _ = run_compacted(
        "default_standard_only_rejects_compacted", None, lifted_fns["op_xrstor64_avx_xsaveopt64_avx"], compacted_avx, dummy
    )
    assert def_comp_faulted, "standard-only default path must #GP on compacted header"
    got_l_comp = invoke_lifted(canonical_fns["op_xrstor64_avx_xsaveopt64_avx"], compacted_avx, dummy)
    assert struct.unpack_from("<I", got_l_comp, 24)[0] == 0x1F80
    got_l_std = invoke_lifted(canonical_fns["op_xrstor64_avx_xsaveopt64_avx"], standard_avx, dummy)
    assert struct.unpack_from("<I", got_l_std, 24)[0] == (0x5F80 & mxcsr_mask)
    assert_equal_output("default_vs_canonical_std", "op_xrstor64_avx_xsaveopt64_avx", 0, got_l_def_std, got_l_std)
    got_l_sse = invoke_lifted(compacted_fns["op_xrstor64_sse_compacted_avx_xsaveopt64_avx"], sse_in, compacted_avx)
    assert struct.unpack_from("<I", got_l_sse, 24)[0] == (0x5F80 & mxcsr_mask)
    if "op_xrstor64_avx_xsaveopt64_avx" in native_fns:
        if has_xsavec:
            got_n_comp = invoke_native(
                harness, native_fns["op_xrstor64_avx_xsaveopt64_avx"], compacted_avx, dummy, entry_ptr=sse_ptr
            )
            assert struct.unpack_from("<I", got_n_comp, 24)[0] == (0x3F80 & mxcsr_mask)
            assert got_n_comp != got_l_comp
        got_n_std = invoke_native(
            harness, native_fns["op_xrstor64_avx_xsaveopt64_avx"], standard_avx, dummy, entry_ptr=sse_ptr
        )
        assert struct.unpack_from("<I", got_n_std, 24)[0] == (0x5F80 & mxcsr_mask)
        assert_equal_output("native_default_std", "op_xrstor64_avx_xsaveopt64_avx", 0, got_n_std, got_l_def_std)
        assert_equal_output("native", "op_xrstor64_avx_xsaveopt64_avx", 0, got_n_std, got_l_std)
    if "op_xrstor64_sse_compacted_avx_xsaveopt64_avx" in native_fns:
        got_n_sse = invoke_native(
            harness, native_fns["op_xrstor64_sse_compacted_avx_xsaveopt64_avx"], sse_in, compacted_avx, entry_ptr=sse_ptr
        )
        assert struct.unpack_from("<I", got_n_sse, 24)[0] == (0x5F80 & mxcsr_mask)
        assert_equal_output("native", "op_xrstor64_sse_compacted_avx_xsaveopt64_avx", 0, got_n_sse, got_l_sse)

    _ = sse_buf
    _ = part_buf


def sdm_xsavec_header(image: bytes, rfbm: int, mxcsr_mask: int) -> tuple[int, bytes]:
    """XSTATE_BV and the MXCSR words xsavec must write after xrstor of image, straight from SDM vol.1 13.6.

    Written without XStateModel so it is an independent oracle: x87 and AVX are in use when their restore bit is
    set, and SSE is in use when its restore bit is set or MXCSR is not the reset value 0x1F80.
    """
    restore_bv = struct.unpack_from("<Q", image, 512)[0]
    mxcsr = struct.unpack_from("<I", image, 24)[0]
    in_use = restore_bv & 5
    if restore_bv & 2 or mxcsr != DEFAULT_MXCSR:
        in_use |= 2
    used = in_use & rfbm
    words = struct.pack("<II", mxcsr, mxcsr_mask) if used & 2 else FILL[24:32]
    return used, words


def check_xsavec_mxcsr_liveness(harness, native: dict, lifted: dict, mxcsr_mask: int, deprecates_cs_ds: bool) -> int:
    """Non-default MXCSR keeps SSE in use for xsavec even when the restore image has XSTATE_BV[1] clear."""
    checks = 0
    for name, rfbm in (("op_xrstor64_xsavec64", 7), ("op_xrstor64_xsavec64_sse", 2)):
        for idx, (a, b) in enumerate(xsave_pairs(mxcsr_mask)):
            used, words = sdm_xsavec_header(a, rfbm, mxcsr_mask)
            got = invoke_lifted(lifted[name], a, b)
            assert struct.unpack_from("<Q", got, 512)[0] == used, f"lifted {name} case {idx}: XSTATE_BV word"
            assert got[24:32] == words, f"lifted {name} case {idx}: MXCSR word"
            if name in native:
                got_n = invoke_native(harness, native[name], a, b)
                got_bv = struct.unpack_from("<Q", got_n, 512)[0]
                assert got_bv & used == used, f"native {name} case {idx}: XSTATE_BV {got_bv:#x} lacks {used:#x}"
                if used & 2:
                    assert got_n[24:32] == words, f"native {name} case {idx}: MXCSR word"
            checks += 1

    # Mutation control: a reference that ignores MXCSR must fail at the MXCSR word (byte 24, the first lost state)
    # and also differ in the XSTATE_BV word (byte 512), for lifted and native.
    a, b = xsave_pairs(mxcsr_mask)[7]
    name = "op_xrstor64_xsavec64"
    assert sdm_xsavec_header(a, 7, mxcsr_mask)[0] == 2, "control needs XSTATE_BV[1] clear in the image and MXCSR non-default"
    right = reference(name, a, b, mxcsr_mask, deprecates_cs_ds)
    model = XStateModel(mxcsr_mask, deprecates_cs_ds)
    model.xrstor(a, rfbm=7, is64=True)
    blind = bytearray(FILL)
    model.xsavec(blind, 0, rfbm=7, is64=True, track_mxcsr=False)
    assert right[512:520] != bytes(blind)[512:520], "MXCSR-blind reference must also lose XSTATE_BV[1]"
    for label in ("lifted", "native"):
        try:
            assert_equal_output(label, name, 7, right, bytes(blind))
        except AssertionError as err:
            if "first mismatch at byte 24:" not in str(err):
                raise AssertionError(f"MXCSR-blind reference failed at the wrong word: {err}") from err
        else:
            raise AssertionError(f"{label} comparison accepted a reference that ignores non-default MXCSR")
    return checks + 2


def sdm_xsaveopt_layout(image: bytes, rfbm: int, mxcsr_mask: int) -> tuple[int, list[tuple[int, int]], bytes]:
    """XSTATE_BV, the spans xsaveopt may write and the MXCSR words after xrstor of image, straight from the SDM.

    Written without XStateModel so it is an independent oracle: x87 and AVX are in use when their restore bit is set,
    SSE is in use when its restore bit is set or MXCSR is not the reset value, and a component that is not in use
    is not written. MXCSR and MXCSR_MASK are stored whenever RFBM selects SSE or AVX.
    """
    restore_bv = struct.unpack_from("<Q", image, 512)[0]
    mxcsr = struct.unpack_from("<I", image, 24)[0]
    in_use = restore_bv & 5
    if restore_bv & 2 or mxcsr != DEFAULT_MXCSR:
        in_use |= 2
    used = in_use & rfbm
    spans = [(512, 520)]
    for bit, parts in XSAVEOPT_COMPONENTS:
        if used & bit:
            spans.extend(parts)
    words = struct.pack("<II", mxcsr, mxcsr_mask) if rfbm & 6 else FILL[24:32]
    if rfbm & 6:
        spans.append((24, 32))
    old_bv = struct.unpack_from("<Q", FILL, 512)[0]
    return (old_bv & ~rfbm) | used, spans, words


def check_xsaveopt_layout(harness, native: dict, lifted: dict, mxcsr_mask: int) -> int:
    """Readback of every byte xsaveopt may touch: the lifted store writes exactly the SDM layout, nothing else."""
    checks = 0
    for name in XSAVEOPT_EXPORTS:
        rfbm = EXPORT_MODES[name][2]
        for idx, (a, b) in enumerate(xsave_pairs(mxcsr_mask)):
            bv, spans, words = sdm_xsaveopt_layout(a, rfbm, mxcsr_mask)
            got = invoke_lifted(lifted[name], a, b)
            assert struct.unpack_from("<Q", got, 512)[0] == bv, f"lifted {name} case {idx}: XSTATE_BV word"
            assert got[24:32] == words, f"lifted {name} case {idx}: MXCSR word"
            for pos in range(BUF_LEN):
                if not any(lo <= pos < hi for lo, hi in spans):
                    assert got[pos] == FILL[pos], f"lifted {name} case {idx}: byte {pos} written outside the layout"
            if name in native:
                got_n = invoke_native(harness, native[name], a, b)
                host_bv = struct.unpack_from("<Q", got_n, 512)[0]
                mxcsr_only = not struct.unpack_from("<Q", a, 512)[0] & 2
                required = bv & rfbm & ~(2 if mxcsr_only else 0)
                assert host_bv & required == required, f"native {name} case {idx}: XSTATE_BV {host_bv:#x}"
                if rfbm & 6:
                    assert got_n[24:32] == words, f"native {name} case {idx}: MXCSR word"
            checks += 1
    return checks


def check_xsaveopt_controls(mxcsr_mask: int, deprecates_cs_ds: bool) -> int:
    """Mutation controls: each wrong lowering or comparator shortcut must fail at a precise word."""
    name = "op_xrstor64_xsaveopt64"
    all_a, all_b = xsave_pairs(mxcsr_mask)[6]
    only_avx_a, only_avx_b = xsave_pairs(mxcsr_mask)[4]
    live_a, live_b = xsave_pairs(mxcsr_mask)[7]
    right = reference(name, all_a, all_b, mxcsr_mask, deprecates_cs_ds)
    right_avx = reference(name, only_avx_a, only_avx_b, mxcsr_mask, deprecates_cs_ds)
    right_live = reference(name, live_a, live_b, mxcsr_mask, deprecates_cs_ds)
    controls = 0

    def must_fail(label, got, want, expect_byte=None):
        nonlocal controls
        for side in ("lifted", "native"):
            try:
                assert_equal_output(side, name, label, got, want)
            except AssertionError as err:
                if expect_byte is not None and f"first mismatch at byte {expect_byte}:" not in str(err):
                    raise AssertionError(f"{label} failed at the wrong word: {err}") from err
            else:
                raise AssertionError(f"{side} comparison accepted mutation {label}")
            controls += 1

    # Active x87, SSE and AVX payload and XSTATE_BV flips fail for lifted and native alike.
    for pos in (0, 32, 160, 576, 512):
        flipped = bytearray(right)
        flipped[pos] ^= 0x01
        must_fail(f"flip{pos}", bytes(flipped), right, pos)
    # Standard xsave (every component stored) instead of xsaveopt: the unused components differ at the first one.
    model = XStateModel(mxcsr_mask, deprecates_cs_ds)
    model.xrstor(only_avx_a, rfbm=7, is64=True)
    plain = bytearray(FILL)
    model.xsave(plain, 0, rfbm=7, is64=True)
    assert bytes(plain) != right_avx, "control needs components that are not in use"
    must_fail("xsave_instead_of_xsaveopt", bytes(plain), right_avx)
    # XSTATE_BV that keeps XINUSE bits outside RFBM: the sse-only request must not report x87 or AVX.
    sse_name = "op_xrstor64_xsaveopt64_sse"
    right_sse = reference(sse_name, all_a, all_b, mxcsr_mask, deprecates_cs_ds)
    unmasked = bytearray(right_sse)
    struct.pack_into("<Q", unmasked, 512, struct.unpack_from("<Q", right_sse, 512)[0] | 5)
    for side in ("lifted", "native"):
        try:
            assert_equal_output(side, sse_name, "bv_without_rfbm", bytes(unmasked), right_sse)
        except AssertionError as err:
            assert "first mismatch at byte 512:" in str(err), err
        else:
            raise AssertionError(f"{side} comparison accepted XSTATE_BV that ignores RFBM")
        controls += 1
    # A reference that ignores non-default MXCSR loses the SSE slot: lifted must fail at the XMM bytes.
    model = XStateModel(mxcsr_mask, deprecates_cs_ds)
    model.xrstor(live_a, rfbm=7, is64=True)
    blind = bytearray(FILL)
    model.xsaveopt(blind, 0, rfbm=7, is64=True, track_mxcsr=False)
    assert right_live[512:520] != bytes(blind)[512:520], "MXCSR-blind reference must also lose XSTATE_BV[1]"
    try:
        assert_equal_output("lifted", name, "mxcsr_blind", right_live, bytes(blind))
    except AssertionError as err:
        assert "first mismatch at byte 160:" in str(err), err
    else:
        raise AssertionError("lifted comparison accepted a reference that ignores non-default MXCSR")
    controls += 1
    # Native may store an initialized component the reference leaves out, but not live bytes in its place.
    host_init = bytearray(right_avx)
    struct.pack_into("<Q", host_init, 512, struct.unpack_from("<Q", right_avx, 512)[0] | 3)
    host_init[0:24] = INIT_X87_HEADER
    host_init[32:416] = bytes(384)
    assert_equal_output("native", name, "host_init", bytes(host_init), right_avx)
    host_live = bytearray(host_init)
    host_live[32] = 0x7F
    try:
        assert_equal_output("native", name, "host_live", bytes(host_live), right_avx)
    except AssertionError:
        controls += 1
    else:
        raise AssertionError("native comparison accepted live bytes in a component the reference leaves out")
    # A host that omits SSE held only by a non-default MXCSR (zero XMM) passes; one that omits live XMM does not.
    host_omit = bytearray(right_live)
    host_omit[160:416] = FILL[160:416]
    struct.pack_into("<Q", host_omit, 512, struct.unpack_from("<Q", right_live, 512)[0] & ~2)
    assert_equal_output("native", name, "host_omit_mxcsr_sse", bytes(host_omit), right_live)
    omit_live = bytearray(right)
    omit_live[160:416] = FILL[160:416]
    struct.pack_into("<Q", omit_live, 512, struct.unpack_from("<Q", right, 512)[0] & ~2)
    try:
        assert_equal_output("native", name, "omit_live_xmm", bytes(omit_live), right)
    except AssertionError:
        controls += 1
    else:
        raise AssertionError("native comparison accepted a host that omitted live XMM state")
    return controls + 1


# --- Compacted-format XRSTOR ---

def reference_compacted(
    name: str, a: bytes, b: bytes, mxcsr_mask: int, deprecates_cs_ds: bool, altered: bool = False, **mutations
) -> bytes:
    """Expected output of a compacted-XRSTOR export; ValueError where the instruction raises #GP."""
    out = bytearray(FILL)
    model = XStateModel(mxcsr_mask, deprecates_cs_ds)
    model.xrstor(a, rfbm=7, is64=True)
    if name == COMPACTED_DYNMASK_EXPORT:
        mask_lo, mask_hi = struct.unpack_from("<II", b, 832)[0], struct.unpack_from("<I", b, 840)[0]
        if mask_hi or mask_lo & ~7:
            raise ValueError("request mask outside x87|SSE|AVX")
        model.xrstor(b, rfbm=mask_lo, is64=True, **mutations)
        model.xsave(out, 0, rfbm=7, is64=True)
        return bytes(out)
    save_rfbm, restore_rfbm = COMPACTED_ROUNDTRIPS[name]
    out[528:576] = bytes(48)  # xsavec leaves the reserved header bytes alone and the restore rejects them
    model.xsavec(out, 0, rfbm=save_rfbm, is64=True)
    model.xrstor(b, rfbm=7, is64=True)
    if altered and name == "op_xsavec_xrstor_all":
        model.xrstor(b, rfbm=7, is64=True)
    else:
        if altered:
            restore_rfbm = 6
        model.xrstor(bytes(out), rfbm=restore_rfbm, is64=name != "op_xsavec_xrstor_x87_avx", **mutations)
    model.xsave(out, 0, rfbm=7, is64=True)
    return bytes(out)


def compact_state_pairs(mxcsr_mask: int):
    """(input_a, input_b) standard images: the state saved with xsavec, then a second live state that replaces it.

    FIP, FDP and FOP stay zero so host x87 pointer quirks cannot show, and the two MXCSR values differ.
    """
    specs = [
        (7, 0x5F80, 31), (7, 0x1F80, 32), (3, 0x3F80, 33), (5, 0x1F80, 34), (6, 0x9FC0, 35),
        (4, 0x1F80, 36), (2, 0x7FC0, 37), (1, 0x1F80, 38), (0, 0x3F80, 39), (0, 0x1F80, 40),
    ]
    pairs = []
    for xbv, mxcsr, seed in specs:
        image_a = make_xsave_image(mxcsr_mask, xstate_bv=xbv, fcw=0x027F, fop=0, mxcsr=mxcsr, seed=seed)
        image_b = make_xsave_image(mxcsr_mask, xstate_bv=7, fcw=0x0F7F, fop=0, mxcsr=0x7F80, seed=seed + 100)
        pairs.append((image_a, image_b))
    return pairs


def compacted_image(
    mxcsr_mask: int, xstate_bv: int, xcomp_bv: int, rfbm: int, *, mxcsr: int = 0x5F80, seed: int = 7, pokes=()
) -> bytes:
    """Hand-built compacted image for the dynmask export: header, payloads at their compacted slots, RFBM at [832]."""
    buf = bytearray(
        make_xsave_image(mxcsr_mask, xstate_bv=xstate_bv, fcw=0x0F7F, fop=0, mxcsr=mxcsr, seed=seed)
    )
    struct.pack_into("<QQ", buf, 512, xstate_bv, xcomp_bv)
    struct.pack_into("<QQ", buf, 832, rfbm, 0)
    for offset, value in pokes:
        buf[offset] = value
    return bytes(buf)


COMPACTED_BIT = 1 << 63
USABLE_XCOMP_HEADER_BITS = range(3, 63)
BAD_MXCSR = 0x11F80


def select_host_disabled_xcomp_bit(host_xcr0: int) -> int:
    """Return the lowest component bit in XCOMP_BV[62:3] that host_xcr0 disables."""
    if (host_xcr0 & 7) != 7:
        raise ValueError(f"host XCR0 lacks x87|SSE|AVX: {host_xcr0:#x}")
    for bit in USABLE_XCOMP_HEADER_BITS:
        if not (host_xcr0 & (1 << bit)):
            return bit
    raise ValueError(f"host XCR0 enables every usable XCOMP_BV bit 3..62: {host_xcr0:#x}")


def native_compacted_header_must_fault(image: bytes, host_xcr0: int, mxcsr_mask: int) -> bool:
    """Return True iff native XRSTOR must #GP on a guest-rejected compacted/standard header under host_xcr0.

    The lifted profile always rejects XCOMP_BV bits outside x87|SSE|AVX (0x7). Native XRSTOR only #GPs on
    XCOMP_BV bits disabled in host_xcr0, so a fixed guest-unsupported case whose listed components are all
    enabled on the host (for example MPX bits 3/4 when XCR0=0x1f) is checked on lifted code only.
    """
    xstate_bv, xcomp_bv = struct.unpack_from("<QQ", image, 512)
    if not (xcomp_bv >> 63):
        return True
    listed = xcomp_bv & ~COMPACTED_BIT
    if any(image[528:576]) or (xstate_bv & ~listed):
        return True
    rfbm = struct.unpack_from("<I", image, 832)[0]
    if (xstate_bv & 2) and (rfbm & 2):
        mxcsr = struct.unpack_from("<I", image, 24)[0]
        if mxcsr & ~mxcsr_mask:
            return True
    return bool(listed & ~host_xcr0)


def compacted_cases(mxcsr_mask: int):
    """(label, input_a, input_b) for the dynmask export: the full valid matrix and every invalid header field."""
    base = make_xsave_image(mxcsr_mask, xstate_bv=7, fcw=0x027F, fop=0, mxcsr=0x3F80, seed=50)
    cases = []
    for listed in range(8):
        for rfbm in range(8):
            for xstate_bv in range(8):
                if xstate_bv & ~listed:
                    continue
                cases.append((
                    f"xcomp{listed}-rfbm{rfbm}-xstate{xstate_bv}",
                    base,
                    compacted_image(mxcsr_mask, xstate_bv, COMPACTED_BIT | listed, rfbm, seed=60 + listed),
                ))
    invalid = (
        ("xcomp-bit3", 0, COMPACTED_BIT | 7 | 8, 7, ()),
        ("xcomp-bit4", 0, COMPACTED_BIT | 7 | 0x10, 7, ()),
        ("xcomp-bit9", 0, COMPACTED_BIT | 7 | (1 << 9), 7, ()),
        ("xcomp-bit62", 0, COMPACTED_BIT | 7 | (1 << 62), 7, ()),
        ("xcomp-bit3-only", 0, COMPACTED_BIT | 8, 0, ()),
        ("xstate-avx-not-in-xcomp", 4, COMPACTED_BIT | 3, 7, ()),
        ("xstate-sse-not-in-xcomp", 2, COMPACTED_BIT | 5, 7, ()),
        ("xstate-x87-not-in-xcomp", 1, COMPACTED_BIT | 6, 7, ()),
        ("xstate-x87-xcomp-empty", 1, COMPACTED_BIT, 7, ()),
        ("xstate-subset-violation-rfbm0", 7, COMPACTED_BIT | 6, 0, ()),
        ("xstate-bit3", 8, COMPACTED_BIT | 7, 7, ()),
        ("xstate-bit63", 1 << 63, COMPACTED_BIT | 7, 7, ()),
        ("reserved-528", 0, COMPACTED_BIT | 7, 7, ((528, 1),)),
        ("reserved-536", 0, COMPACTED_BIT | 7, 7, ((536, 0x80),)),
        ("reserved-552", 0, COMPACTED_BIT | 7, 7, ((552, 1),)),
        ("reserved-575", 0, COMPACTED_BIT | 7, 7, ((575, 1),)),
        ("standard-xcomp-sse", 0, 2, 7, ()),
        ("standard-xcomp-bit62", 0, 1 << 62, 7, ()),
    )
    for label, xstate_bv, xcomp_bv, rfbm, pokes in invalid:
        cases.append((label, base, compacted_image(mxcsr_mask, xstate_bv, xcomp_bv, rfbm, pokes=pokes)))
    # MXCSR is checked only when it is loaded: SSE restored from the image, never the init, untouched or absent cases.
    mxcsr_pokes = tuple((24 + i, (BAD_MXCSR >> (8 * i)) & 0xFF) for i in range(4))
    for label, xstate_bv, listed, rfbm in (
        ("bad-mxcsr-sse-restored", 2, 7, 7),
        ("bad-mxcsr-sse-only", 2, 7, 2),
        ("bad-mxcsr-sse-init", 0, 7, 7),
        ("bad-mxcsr-avx-only-request", 4, 7, 4),
        ("bad-mxcsr-sse-not-listed", 1, 5, 7),
        ("bad-mxcsr-x87-only-request", 3, 7, 1),
    ):
        cases.append((
            label, base, compacted_image(mxcsr_mask, xstate_bv, COMPACTED_BIT | listed, rfbm, pokes=mxcsr_pokes)
        ))
    return cases


def _run_compacted_raw(label: str, harness, fn, a: bytes, b: bytes, *, b_offset: int = 0):
    """Call one export (harness is None for lifted code); returns (faulted, exception_text, output_bytes)."""
    buf_a, ptr_a, a_pre, a_off, a_post = alloc_aligned_buffer(a)
    buf_b, ptr_b, b_pre, b_off, b_post = alloc_aligned_buffer(b, offset=b_offset)
    buf_out, ptr_out, o_pre, o_off, o_post = alloc_aligned_buffer(FILL)
    faulted = False
    exc_text: str | None = None
    try:
        if harness is not None:
            harness.call_native(fn, ptr_a, ptr_b, ptr_out)
        else:
            fn(ptr_a, ptr_b, ptr_out)
    except OSError as exc:
        assert GP_MESSAGE in str(exc).lower(), (label, exc)
        faulted = True
        exc_text = str(exc)
    check_guards_and_read(buf_a, a_pre, a_off, a_post, f"{label} input_a")
    check_guards_and_read(buf_b, b_pre, b_off, b_post, f"{label} input_b")
    return faulted, exc_text, check_guards_and_read(buf_out, o_pre, o_off, o_post, f"{label} output")


def run_compacted(label: str, harness, fn, a: bytes, b: bytes, *, b_offset: int = 0):
    """Call one export (harness is None for lifted code); returns (faulted with #GP, output bytes)."""
    faulted, _, out = _run_compacted_raw(label, harness, fn, a, b, b_offset=b_offset)
    return faulted, out


def expected_compacted(*args, **kwargs):
    """The reference output, or None where the instruction raises #GP."""
    try:
        return reference_compacted(*args, **kwargs)
    except ValueError:
        return None


def check_compacted_run(side: str, name: str, label, harness, fn, a: bytes, b: bytes, want, *, b_offset: int = 0) -> None:
    faulted, got = run_compacted(f"{side} {name} {label}", harness, fn, a, b, b_offset=b_offset)
    if want is None:
        assert faulted, f"{side} {name} {label} accepted a header the instruction rejects"
        assert got == FILL, f"{side} {name} {label}: output written before the #GP"
    else:
        assert not faulted, f"{side} {name} {label} raised #GP for a valid compacted restore"
        want_bv = struct.unpack_from("<Q", want, 512)[0]
        got_bv = struct.unpack_from("<Q", got, 512)[0]
        if (
            side == "native"
            and want_bv & 2
            and not got_bv & 2
            and want[160:416] == bytes(256)
            and struct.unpack_from("<I", want, 24)[0] != DEFAULT_MXCSR
        ):
            # A host may report XSTATE_BV[1] clear for SSE held only by a non-default MXCSR over zero XMM state,
            # as for xsaveopt; the XMM bytes and the MXCSR word stay compared.
            want = want[:512] + struct.pack("<Q", want_bv & ~2) + want[520:]
        assert_equal_output(side, name, label, got, want)


def check_host_xcr0_selection_controls(lifted: dict, cases, mxcsr_mask: int, deprecates_cs_ds: bool) -> int:
    """Synthetic host-XCR0 controls (including MPX XCR0=0x1f) so non-MPX hosts still cover the bit selection."""
    controls = 0
    for synthetic_xcr0, expected_bit in (
        (0x7, 3),
        (0x1F, 5),
        (0xE7, 3),
        (0xFF, 8),
        (0x2FF, 8),
        (0x3FF, 10),
        ((1 << 62) - 1, 62),
    ):
        picked = select_host_disabled_xcomp_bit(synthetic_xcr0)
        assert picked == expected_bit, (hex(synthetic_xcr0), picked, expected_bit)
        assert not (synthetic_xcr0 & (1 << picked)), (hex(synthetic_xcr0), picked)
        assert picked in USABLE_XCOMP_HEADER_BITS, picked
        controls += 1

    for bad_xcr0 in (0x0, 0x3, (1 << 63) - 1):
        try:
            select_host_disabled_xcomp_bit(bad_xcr0)
        except ValueError:
            controls += 1
        else:
            raise AssertionError(f"select_host_disabled_xcomp_bit accepted {bad_xcr0:#x}")

    # Synthetic MPX host (XCR0=0x1f: x87, SSE, AVX, BNDREGS, BNDCSR) and PKRU host (XCR0=0x2e7):
    # fixed guest-unsupported XCOMP_BV bits enabled by that host stay refused on lifted code, while the native
    # fault predicate excludes them and keeps every other invalid header (including xstate-bit3 subset violations).
    mpx_xcr0 = 0x1F
    pkru_xcr0 = 0x2E7
    by_label = {label: (a, b) for label, a, b in cases}
    mpx_native_accepted = {"xcomp-bit3", "xcomp-bit4", "xcomp-bit3-only"}
    for label, a, b in cases:
        want = expected_compacted(COMPACTED_DYNMASK_EXPORT, a, b, mxcsr_mask, deprecates_cs_ds)
        if want is not None:
            continue
        must_fault_mpx = native_compacted_header_must_fault(b, mpx_xcr0, mxcsr_mask)
        if label in mpx_native_accepted:
            assert not must_fault_mpx, f"MPX host XCR0=0x1f must not require native #GP on {label}"
        else:
            assert must_fault_mpx, f"MPX host XCR0=0x1f must still require native #GP on {label}"
        controls += 1
    assert not native_compacted_header_must_fault(by_label["xcomp-bit9"][1], pkru_xcr0, mxcsr_mask)
    assert native_compacted_header_must_fault(by_label["xcomp-bit3"][1], pkru_xcr0, mxcsr_mask)
    controls += 2

    # Mutation control for XCR0=0x1f: picking bit 3 (the old hardcoded assumption) fails the native-fault gate,
    # whereas the selected disabled bit (bit 5) passes both the native-fault gate and the lifted DLL refusal.
    mpx_bit = select_host_disabled_xcomp_bit(mpx_xcr0)
    assert mpx_bit == 5
    base = make_xsave_image(mxcsr_mask, xstate_bv=7, fcw=0x027F, fop=0, mxcsr=0x3F80, seed=50)
    mpx_fault_img = compacted_image(mxcsr_mask, 7, COMPACTED_BIT | 7 | (1 << mpx_bit), 7, seed=67)
    mutated_bit3_img = compacted_image(mxcsr_mask, 7, COMPACTED_BIT | 7 | (1 << 3), 7, seed=67)
    assert native_compacted_header_must_fault(mpx_fault_img, mpx_xcr0, mxcsr_mask)
    assert not native_compacted_header_must_fault(mutated_bit3_img, mpx_xcr0, mxcsr_mask)
    assert expected_compacted(COMPACTED_DYNMASK_EXPORT, base, mpx_fault_img, mxcsr_mask, deprecates_cs_ds) is None
    check_compacted_run("lifted", COMPACTED_DYNMASK_EXPORT, f"mpx-sim-bit{mpx_bit}", None,
                        lifted[COMPACTED_DYNMASK_EXPORT], base, mpx_fault_img, None)
    controls += 2
    return controls


def check_compacted_xrstor(
    harness, native: dict, lifted: dict, altered: dict, mxcsr_mask: int, deprecates_cs_ds: bool, host_xcr0: int
) -> dict:
    """XSAVEC -> XRSTOR round trips, hand-built compacted headers, their #GP rules and the mutation controls."""
    counts = {
        "roundtrip": 0,
        "header": 0,
        "gp": 0,
        "altered": {},
        "controls": 0,
        "host_disabled_bit": select_host_disabled_xcomp_bit(host_xcr0),
        "native_xcomp_bit": None,
        "native_xcomp_exc": None,
        "lifted_only_xcomp": [],
        "xcr0_controls": 0,
    }

    def model(name, a, b, **mutations):
        return expected_compacted(name, a, b, mxcsr_mask, deprecates_cs_ds, **mutations)

    for name in COMPACTED_ROUNDTRIPS:
        for idx, (a, b) in enumerate(compact_state_pairs(mxcsr_mask)):
            want = model(name, a, b)
            assert want is not None
            check_compacted_run("lifted", name, idx, None, lifted[name], a, b, want)
            if name in native:
                check_compacted_run("native", name, idx, harness, native[name], a, b, want)
            counts["roundtrip"] += 1
            if name == "op_xsavec_xrstor_all":
                # Independent oracle: a full save and restore of the same state brings back exactly what a direct
                # restore of the image gives, in the legacy area and the AVX area.
                direct = reference("op_xrstor64_xsave64", a, a, mxcsr_mask, deprecates_cs_ds)
                assert want[:512] == direct[:512] and want[576:832] == direct[576:832], f"round trip case {idx}"

    cases = compacted_cases(mxcsr_mask)
    for label, a, b in cases:
        want = model(COMPACTED_DYNMASK_EXPORT, a, b)
        check_compacted_run("lifted", COMPACTED_DYNMASK_EXPORT, label, None, lifted[COMPACTED_DYNMASK_EXPORT], a, b, want)
        if COMPACTED_DYNMASK_EXPORT in native:
            if want is not None or native_compacted_header_must_fault(b, host_xcr0, mxcsr_mask):
                check_compacted_run(
                    "native", COMPACTED_DYNMASK_EXPORT, label, harness, native[COMPACTED_DYNMASK_EXPORT], a, b, want
                )
            else:
                counts["lifted_only_xcomp"].append(label)
        counts["header"] += want is not None
        counts["gp"] += want is None

    # Host-XCR0-derived native fault comparison: keep the header and request identical to a valid compacted restore
    # control except for one XCOMP_BV component bit in [3..62] that the measured host XCR0 actually disables.
    disabled_bit = counts["host_disabled_bit"]
    base = make_xsave_image(mxcsr_mask, xstate_bv=7, fcw=0x027F, fop=0, mxcsr=0x3F80, seed=50)
    valid_ctrl = compacted_image(mxcsr_mask, 7, COMPACTED_BIT | 7, 7, seed=67)
    host_fault = compacted_image(mxcsr_mask, 7, COMPACTED_BIT | 7 | (1 << disabled_bit), 7, seed=67)
    host_fault_rfbm0 = compacted_image(mxcsr_mask, 0, COMPACTED_BIT | (1 << disabled_bit), 0, seed=67)
    assert host_fault[:520] == valid_ctrl[:520] and host_fault[528:] == valid_ctrl[528:]
    assert (struct.unpack_from("<Q", host_fault, 520)[0] ^ struct.unpack_from("<Q", valid_ctrl, 520)[0]) == (1 << disabled_bit)
    want_valid_ctrl = model(COMPACTED_DYNMASK_EXPORT, base, valid_ctrl)
    assert want_valid_ctrl is not None
    assert model(COMPACTED_DYNMASK_EXPORT, base, host_fault) is None
    assert model(COMPACTED_DYNMASK_EXPORT, base, host_fault_rfbm0) is None
    check_compacted_run("lifted", COMPACTED_DYNMASK_EXPORT, "host-xcr0-valid-control", None,
                        lifted[COMPACTED_DYNMASK_EXPORT], base, valid_ctrl, want_valid_ctrl)
    check_compacted_run("lifted", COMPACTED_DYNMASK_EXPORT, f"host-xcr0-disabled-bit{disabled_bit}", None,
                        lifted[COMPACTED_DYNMASK_EXPORT], base, host_fault, None)
    check_compacted_run("lifted", COMPACTED_DYNMASK_EXPORT, f"host-xcr0-disabled-bit{disabled_bit}-rfbm0", None,
                        lifted[COMPACTED_DYNMASK_EXPORT], base, host_fault_rfbm0, None)
    if COMPACTED_DYNMASK_EXPORT in native:
        check_compacted_run("native", COMPACTED_DYNMASK_EXPORT, "host-xcr0-valid-control", harness,
                            native[COMPACTED_DYNMASK_EXPORT], base, valid_ctrl, want_valid_ctrl)
        faulted, exc_text, got = _run_compacted_raw(
            f"native {COMPACTED_DYNMASK_EXPORT} host-xcr0-disabled-bit{disabled_bit}",
            harness, native[COMPACTED_DYNMASK_EXPORT], base, host_fault,
        )
        assert faulted, (
            f"native {COMPACTED_DYNMASK_EXPORT} accepted host-disabled XCOMP_BV bit {disabled_bit} (XCR0={host_xcr0:#x})"
        )
        assert got == FILL, f"native {COMPACTED_DYNMASK_EXPORT} host-disabled bit {disabled_bit}: output written before #GP"
        check_compacted_run("native", COMPACTED_DYNMASK_EXPORT, f"host-xcr0-disabled-bit{disabled_bit}-rfbm0", harness,
                            native[COMPACTED_DYNMASK_EXPORT], base, host_fault_rfbm0, None)
        counts["native_xcomp_bit"] = disabled_bit
        counts["native_xcomp_exc"] = exc_text
    counts["gp"] += 2

    # A misaligned compacted image faults like a misaligned standard one, before anything is stored.
    mis_a, mis_b = compact_state_pairs(mxcsr_mask)[0][0], compacted_image(mxcsr_mask, 7, COMPACTED_BIT | 7, 7)
    for off in (8, 16):
        check_compacted_run("lifted", COMPACTED_DYNMASK_EXPORT, f"misaligned+{off}", None,
                            lifted[COMPACTED_DYNMASK_EXPORT], mis_a, mis_b, None, b_offset=off)
        if COMPACTED_DYNMASK_EXPORT in native:
            check_compacted_run("native", COMPACTED_DYNMASK_EXPORT, f"misaligned+{off}", harness,
                                native[COMPACTED_DYNMASK_EXPORT], mis_a, mis_b, None, b_offset=off)
        counts["gp"] += 1

    # Request masks outside x87|SSE|AVX are this translator's own bound (hardware masks them with XCR0): lifted only.
    for mask_lo, mask_hi in ((8, 0), (0xF, 0), (7, 1)):
        bad_mask_image = bytearray(compacted_image(mxcsr_mask, 0, COMPACTED_BIT | 7, 0))
        struct.pack_into("<QQ", bad_mask_image, 832, mask_lo, mask_hi)
        check_compacted_run("lifted", COMPACTED_DYNMASK_EXPORT, f"mask {mask_hi:#x}:{mask_lo:#x}", None,
                            lifted[COMPACTED_DYNMASK_EXPORT], cases[0][1], bytes(bad_mask_image), None)
        counts["gp"] += 1

    for name in ALTERED_COMPACTED_EXPORTS:
        diffs = 0
        for idx, (a, b) in enumerate(compact_state_pairs(mxcsr_mask)):
            want_alt = model(name, a, b, altered=True)
            _, got_alt = run_compacted(f"altered {name} {idx}", None, altered[name], a, b)
            assert_equal_output("altered", name, idx, got_alt, want_alt)
            diffs += got_alt != model(name, a, b)
        assert diffs, f"altered control {name} did not diverge from original"
        counts["altered"][name] = diffs

    counts["controls"] = check_compacted_controls(lifted, cases, mxcsr_mask, deprecates_cs_ds)
    counts["xcr0_controls"] = check_host_xcr0_selection_controls(lifted, cases, mxcsr_mask, deprecates_cs_ds)
    return counts


def check_compacted_controls(lifted: dict, cases, mxcsr_mask: int, deprecates_cs_ds: bool) -> int:
    """Mutation controls: dropping one validity rule or the compacted MXCSR rule from the model must be caught."""
    by_label = {label: (a, b) for label, a, b in cases}
    dyn = COMPACTED_DYNMASK_EXPORT
    controls = 0

    # Each validity rule has a case only that rule rejects: the model without it accepts the header, the lifted code
    # (and the host) fault, so a lowering that dropped the rule would show up as an accepted header.
    for label, mutation in (
        ("xstate-avx-not-in-xcomp", {"check_subset": False}),
        ("xstate-x87-xcomp-empty", {"check_subset": False}),
        ("reserved-536", {"check_reserved": False}),
        ("reserved-575", {"check_reserved": False}),
        ("xcomp-bit3", {"check_xcomp": False}),
        ("xcomp-bit62", {"check_xcomp": False}),
    ):
        a, b = by_label[label]
        assert expected_compacted(dyn, a, b, mxcsr_mask, deprecates_cs_ds) is None, label
        accepted = expected_compacted(dyn, a, b, mxcsr_mask, deprecates_cs_ds, **mutation)
        assert accepted is not None, f"mutation {mutation} still rejects {label}"
        faulted, _ = run_compacted(f"control {label}", None, lifted[dyn], a, b)
        assert faulted, f"lifted {label}"
        try:
            check_compacted_run("lifted", dyn, label, None, lifted[dyn], a, b, accepted)
        except AssertionError:
            controls += 1
        else:
            raise AssertionError(f"comparison accepted the mutation {mutation} on {label}")

    # The compacted MXCSR rule against the standard one: the first differing byte is inside the MXCSR word [24..27].
    mxcsr_cases = []
    for label in ("xcomp7-rfbm7-xstate0", "xcomp7-rfbm4-xstate4", "xcomp5-rfbm7-xstate1"):
        a, b = by_label[label]
        mxcsr_cases.append((dyn, a, b, label))
    for name in ("op_xsavec_xrstor_avx_only", "op_xsavec_xrstor_x87_avx"):
        a, b = compact_state_pairs(mxcsr_mask)[0]
        mxcsr_cases.append((name, a, b, "roundtrip"))
    for name, a, b, label in mxcsr_cases:
        right = expected_compacted(name, a, b, mxcsr_mask, deprecates_cs_ds)
        standard_rule = expected_compacted(name, a, b, mxcsr_mask, deprecates_cs_ds, compacted_mxcsr=False)
        if right == standard_rule:
            continue  # the rules agree here: the loaded image MXCSR equals what the compacted rule gives
        if standard_rule is None:
            controls += 1  # the standard rule would load a reserved MXCSR the compacted save never wrote and fault
            continue
        _, got = run_compacted(f"control {name} {label}", None, lifted[name], a, b)
        assert_equal_output("lifted", name, label, got, right)
        try:
            assert_equal_output("lifted", name, label, got, standard_rule)
        except AssertionError as err:
            assert any(f"first mismatch at byte {pos}:" in str(err) for pos in range(24, 28)), err
            controls += 1
        else:
            raise AssertionError(f"comparison accepted the standard MXCSR rule on {name} {label}")
    assert controls >= 9, controls
    return controls



def check_differential(
    native_path: Path,
    lifted_path: Path,
    altered_path: Path,
    canonical_path: Path,
    canonical_altered_path: Path,
    native_altered_path: Path,
    compacted_path: Path,
    compacted_altered_path: Path,
) -> None:
    lifted_lib, lifted = bind_exports(
        lifted_path,
        "lifted_",
        DEFINED_EXPORTS + (DYNMASK_EXPORT, XSAVEOPT_DYNMASK_EXPORT, "op_xrstor64_avx_xsaveopt64_avx"),
    )
    lifted_lib.host_has_xsave.argtypes = []
    lifted_lib.host_has_xsave.restype = ctypes.c_uint32
    if not lifted_lib.host_has_xsave():
        print("Host CPU lacks XSAVE/OSXSAVE/AVX (XCR0 & 7 == 7); skipping native differential check.")
        raise SystemExit(77)

    lifted_lib.host_has_xsavec.argtypes = []
    lifted_lib.host_has_xsavec.restype = ctypes.c_uint32
    has_xsavec = bool(lifted_lib.host_has_xsavec())
    lifted_lib.host_has_xsaveopt.argtypes = []
    lifted_lib.host_has_xsaveopt.restype = ctypes.c_uint32
    has_xsaveopt = bool(lifted_lib.host_has_xsaveopt())
    if not has_xsaveopt:
        print("Host CPU lacks XSAVEOPT (CPUID.(0xD,1):EAX[0]); native xsaveopt fixtures are skipped, lifted xsaveopt still checked.")
    if not has_xsavec:
        print("Host CPU lacks XSAVEC (CPUID.(0xD,1):EAX[1]); native xsavec fixtures are skipped, lifted xsavec still checked.")

    lifted_lib.host_xsave_area_size.argtypes = []
    lifted_lib.host_xsave_area_size.restype = ctypes.c_uint32
    host_area_size = int(lifted_lib.host_xsave_area_size())
    lifted_lib.host_xcr0.argtypes = []
    lifted_lib.host_xcr0.restype = ctypes.c_uint64
    host_xcr0 = int(lifted_lib.host_xcr0())
    print(f"Host XSAVE area (CPUID.0D.0:EBX): {host_area_size} bytes, XCR0={host_xcr0:#x}", flush=True)

    lifted_lib.host_deprecates_fpu_cs_ds.argtypes = []
    lifted_lib.host_deprecates_fpu_cs_ds.restype = ctypes.c_uint32
    lifted_lib.host_fp_err_cpuid_info.argtypes = []
    lifted_lib.host_fp_err_cpuid_info.restype = ctypes.c_uint32
    lifted_lib.host_mxcsr_mask.argtypes = []
    lifted_lib.host_mxcsr_mask.restype = ctypes.c_uint32

    deprecates_cs_ds = bool(lifted_lib.host_deprecates_fpu_cs_ds())
    fp_cpuid = int(lifted_lib.host_fp_err_cpuid_info())
    mxcsr_mask = int(lifted_lib.host_mxcsr_mask())

    _, native = bind_exports(native_path, "", EXPORTS + COMPACTED_EXPORTS + (SSE_COMPACTED_CONTROL_EXPORT,))
    native = native_exports_for_host(native, has_xsavec, has_xsaveopt)
    _, native_altered = bind_exports(native_altered_path, "", ALTERED_CANONICAL_EXPORTS)
    altered_lib, altered = bind_exports(altered_path, "lifted_", ALTERED_DEFINED_EXPORTS)
    canonical_lib, canonical = bind_exports(canonical_path, "lifted_", CANONICAL_EXPORTS)
    canonical_altered_lib, canonical_altered = bind_exports(
        canonical_altered_path, "lifted_", ALTERED_CANONICAL_EXPORTS
    )
    compacted_lib, compacted_lifted = bind_exports(
        compacted_path, "lifted_", COMPACTED_EXPORTS + (SSE_COMPACTED_CONTROL_EXPORT,)
    )
    compacted_altered_lib, compacted_altered = bind_exports(
        compacted_altered_path, "lifted_", ALTERED_COMPACTED_EXPORTS
    )

    harness = NativeHarness(host_area_size, mxcsr_mask)
    try:
        fp_caps = harness.probe_xsave_err_ptrs(mxcsr_mask)
        check_entry_state_contracts(
            harness,
            native,
            native_altered,
            lifted_lib,
            lifted,
            altered_lib,
            canonical_lib,
            canonical,
            canonical_altered_lib,
            canonical_altered,
            compacted_lib,
            compacted_lifted,
            compacted_altered_lib,
            mxcsr_mask,
            deprecates_cs_ds,
            has_xsavec,
        )
        misalign_checks, hdr_mxcsr_checks = check_fault_controls(
            harness, native, lifted, canonical, mxcsr_mask
        )
        check_discarded_read(harness, native_path, lifted_path, mxcsr_mask, deprecates_cs_ds, fp_caps)
        check_lifted_preserves_host_state(harness, lifted, mxcsr_mask)
        mask_checks = check_xsavec_dynamic_mask(lifted[DYNMASK_EXPORT], mxcsr_mask, deprecates_cs_ds)
        mask_checks += check_xsavec_dynamic_mask(
            lifted[XSAVEOPT_DYNMASK_EXPORT], mxcsr_mask, deprecates_cs_ds, XSAVEOPT_DYNMASK_EXPORT, "xsaveopt"
        )

        total = 0
        for name in EXPORTS:
            pairs = disp_pairs(mxcsr_mask) if name == "op_xrstor64_disp" else xsave_pairs(mxcsr_mask)
            lifted_fn = canonical[name] if name in CANONICAL_EXPORTS else lifted[name]
            for idx, (a, b) in enumerate(pairs):
                want = reference(name, a, b, mxcsr_mask, deprecates_cs_ds)
                want_native = (
                    want
                    if name in CANONICAL_EXPORTS
                    else expected_native_output(want, a, name, fp_caps)
                )
                got_l = invoke_lifted(lifted_fn, a, b)
                if name in native:
                    got_n = invoke_native(harness, native[name], a, b)
                    assert_equal_output("native", name, idx, got_n, want_native)
                assert_equal_output("lifted", name, idx, got_l, want)
                total += 1

        # Negative control: XSAVE with RFBM = AVX alone stores MXCSR, so output that leaves it out must be caught.
        a, b = xsave_pairs(mxcsr_mask)[0]
        want = reference(AVX_XSAVE_EXPORT, a, b, mxcsr_mask, deprecates_cs_ds)
        try:
            assert_equal_output("lifted", AVX_XSAVE_EXPORT, 0, want[:24] + FILL[24:32] + want[32:], want)
        except AssertionError:
            pass
        else:
            raise AssertionError("XSAVE output without MXCSR was not caught by the comparator")
        # Negative control: an MM read that leaves TOP, the tags and XINUSE[0] alone must be caught.
        want = reference(MMX_READ_EXPORT, a, b, mxcsr_mask, deprecates_cs_ds)
        unchanged = reference("op_xrstor64_xsave64", a, b, mxcsr_mask, deprecates_cs_ds)
        try:
            assert_equal_output("lifted", MMX_READ_EXPORT, 0, unchanged, want)
        except AssertionError:
            pass
        else:
            raise AssertionError("an MM read that leaves the x87 state alone was not caught by the comparator")

        diverged = {}
        for name in ALTERED_DEFINED_EXPORTS + ALTERED_CANONICAL_EXPORTS:
            pairs = disp_pairs(mxcsr_mask) if name == "op_xrstor64_disp" else xsave_pairs(mxcsr_mask)
            alt_fn = canonical_altered[name] if name in ALTERED_CANONICAL_EXPORTS else altered[name]
            diffs = 0
            for idx, (a, b) in enumerate(pairs):
                want_alt = reference(name, a, b, mxcsr_mask, deprecates_cs_ds, altered=True)
                want_orig = reference(name, a, b, mxcsr_mask, deprecates_cs_ds)
                got_alt = invoke_lifted(alt_fn, a, b)
                assert_equal_output("altered", name, idx, got_alt, want_alt)
                if got_alt != want_orig:
                    diffs += 1
            if not diffs:
                raise AssertionError(f"altered control {name} did not diverge from original")
            diverged[name] = diffs

        # Output mutation negative control
        sample_a, sample_b = xsave_pairs(mxcsr_mask)[1]
        expected = reference("op_xrstor64_xsave64", sample_a, sample_b, mxcsr_mask, deprecates_cs_ds)
        mutated = bytearray(expected)
        mutated[512] ^= 0x04
        try:
            assert_equal_output("mutation", "op_xrstor64_xsave64", 0, bytes(mutated), expected)
        except AssertionError:
            pass
        else:
            raise AssertionError("mutated output was not caught by comparator")

        # XSAVEC word controls: XCOMP_BV[63], XSTATE_BV, a reserved header byte and the AVX slot must each be caught.
        xc_a, xc_b = xsave_pairs(mxcsr_mask)[6]
        xc_name = "op_xrstor64_xsavec64"
        xc_want = reference(xc_name, xc_a, xc_b, mxcsr_mask, deprecates_cs_ds)
        for pos, flip in ((519, 0x80), (512, 0x04), (528, 0x01), (576, 0x01)):
            xc_bad = bytearray(xc_want)
            xc_bad[pos] ^= flip
            try:
                assert_equal_output("mutation", xc_name, pos, bytes(xc_bad), xc_want)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"mutated xsavec byte {pos} was not caught by comparator")

        liveness_checks = check_xsavec_mxcsr_liveness(harness, native, lifted, mxcsr_mask, deprecates_cs_ds)
        layout_checks = check_xsaveopt_layout(harness, native, lifted, mxcsr_mask)
        opt_controls = check_xsaveopt_controls(mxcsr_mask, deprecates_cs_ds)
        compacted = check_compacted_xrstor(
            harness, native, compacted_lifted, compacted_altered, mxcsr_mask, deprecates_cs_ds, host_xcr0
        )

        # Native comparison keeps active x87/SSE payloads: a flipped FCW, ST0 or XMM0 byte must fail, while an
        # initialized component carrying extra host bytes must still pass.
        want_bv = struct.unpack_from("<Q", xc_want, 512)[0]
        assert want_bv & 3 == 3, "native-strictness control needs active x87 and SSE components"
        for pos in (0, 32, 160):
            flipped = bytearray(xc_want)
            flipped[pos] ^= 0x01
            try:
                assert_equal_output("native", xc_name, pos, bytes(flipped), xc_want)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"native comparison accepted a flipped active-component byte {pos}")
        sse_name = "op_xrstor64_xsavec64_sse"
        sse_a, sse_b = xsave_pairs(mxcsr_mask)[6]
        sse_want = reference(sse_name, sse_a, sse_b, mxcsr_mask, deprecates_cs_ds)
        assert struct.unpack_from("<Q", sse_want, 512)[0] & 1 == 0
        sse_host = bytearray(sse_want)
        struct.pack_into("<Q", sse_host, 512, struct.unpack_from("<Q", sse_want, 512)[0] | 1)
        sse_host[0:24] = bytes(range(1, 25))
        sse_host[32:160] = bytes(range(128))
        assert_equal_output("native", sse_name, 0, bytes(sse_host), sse_want)

        # A host without XSAVEC must not be handed native xsavec fixtures.
        gated = native_exports_for_host(native, False, False)
        unsupported = (
            set(XSAVEC_EXPORTS)
            | set(XSAVEOPT_EXPORTS)
            | set(COMPACTED_EXPORTS)
            | {"op_xrstor64_avx_xsaveopt64_avx", SSE_COMPACTED_CONTROL_EXPORT}
        )
        assert not unsupported & set(gated) and set(native) - unsupported == set(gated)
        check_fault_controls(harness, gated, lifted, canonical, mxcsr_mask)
        # XSAVEOPT without XSAVEC (Haswell, Broadwell): the compacted SSE control goes, the general
        # XRSTOR(4) -> XSAVEOPT64(4) export stays for its standard-image comparison.
        gated_opt = native_exports_for_host(native, False, True)
        assert SSE_COMPACTED_CONTROL_EXPORT not in gated_opt and "op_xrstor64_avx_xsaveopt64_avx" in gated_opt
        assert not (set(XSAVEC_EXPORTS) | set(COMPACTED_EXPORTS)) & set(gated_opt)

        es0_64 = fp_caps[(True, True, False)][:3]
        es1_64 = fp_caps[(True, True, True)][:3]
        if compacted["native_xcomp_bit"] is not None:
            native_xcomp_note = (
                f"native host-disabled XCOMP_BV bit {compacted['native_xcomp_bit']} "
                f"(XCR0={host_xcr0:#x}, exc={compacted['native_xcomp_exc']!r}, valid restore control matched)"
            )
            if compacted["lifted_only_xcomp"]:
                native_xcomp_note += f", lifted-only guest XCOMP_BV refusals={compacted['lifted_only_xcomp']}"
        else:
            native_xcomp_note = (
                f"host-disabled XCOMP_BV bit {compacted['host_disabled_bit']} "
                f"(XCR0={host_xcr0:#x}, native skipped: no XSAVEC)"
            )
        print(
            f"{total} native/lifted/reference XSAVE/XRSTOR cases across {len(EXPORTS)} exports matched bit-for-bit "
            f"(mxcsr_mask={mxcsr_mask:#x}, deprecates_fpu_cs_ds={deprecates_cs_ds}, "
            f"fp_cpuid={fp_cpuid:#x}, fp_es0={es0_64}, fp_es1={es1_64}); "
            f"ALTER_OP diverged on {diverged}; {misalign_checks} misalignment #GP checks, "
            f"{hdr_mxcsr_checks} header/MXCSR #GP checks, {mask_checks} unsupported-mask #GP checks, entry-state contract, non-default entry controls, "
            f"dead-read, host-state, and output mutation controls passed; "
            f"{liveness_checks} xsavec MXCSR-liveness checks passed; "
            f"{layout_checks} xsaveopt layout readbacks and {opt_controls} xsaveopt mutation controls passed; "
            f"compacted xrstor: {compacted['roundtrip']} xsavec round trips, {compacted['header']} valid and "
            f"{compacted['gp']} #GP header cases ({native_xcomp_note}), ALTER_OP diverged on {compacted['altered']}, "
            f"{compacted['controls']} mutation controls and {compacted['xcr0_controls']} host-XCR0 controls passed"
        )
    finally:
        harness.close()


# --- Alternate-encoding acceptance & rejection controls ---

def check_alt_encodings_and_rejections(original_path: Path, altered_path: Path) -> None:
    mov_eax_1 = Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EAX, 1)
    mov_eax_2 = Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EAX, 2)
    mov_eax_3 = Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EAX, 3)
    mov_eax_4 = Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EAX, 4)
    mov_eax_7 = Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EAX, 7)
    xor_edx = Instruction.create_reg_reg(Code.XOR_R32_RM32, R.EDX, R.EDX)
    ret_ins = Instruction.create(Code.RETNQ)

    alt_accept = [
        ("xrstor_r9_saved_rdx", (
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R9, R.RDX),
            mov_eax_7,
            xor_edx,
            Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.R9)),
            Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8)),
            ret_ins)),
        ("xrstor3_vzeroupper_xsave7", (
            mov_eax_3,
            xor_edx,
            Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
            Instruction.create(Code.VEX_VZEROUPPER),
            mov_eax_7,
            Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8)),
            ret_ins)),
        ("vmovdqa_ymm_roundtrip", (
            Instruction.create_reg_mem(Code.VEX_VMOVDQA_YMM_YMMM256, R.YMM2, MemoryOperand(R.RCX)),
            Instruction.create_mem_reg(Code.VEX_VMOVDQA_YMMM256_YMM, MemoryOperand(R.R8), R.YMM2),
            ret_ins)),
        ("movdqa_xmm_roundtrip", (
            Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM3, MemoryOperand(R.RCX)),
            Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8), R.XMM3),
            ret_ins)),
        ("movq_mm_gpr_roundtrip", (
            Instruction.create_reg_mem(Code.MOVQ_MM_MMM64, R.MM2, MemoryOperand(R.RCX)),
            Instruction.create_reg_reg(Code.MOVQ_RM64_MM, R.R10, R.MM2),
            Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.R8), R.R10),
            Instruction.create(Code.EMMS),
            ret_ins)),
    ]
    alt_accept.append(("xrstor_xsavec64", (
        mov_eax_7,
        xor_edx,
        Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
        Instruction.create_mem(Code.XSAVEC64_MEM, MemoryOperand(R.R8)),
        ret_ins)))
    alt_accept.append(("xrstor_xsaveopt64", (
        mov_eax_7,
        xor_edx,
        Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
        Instruction.create_mem(Code.XSAVEOPT64_MEM, MemoryOperand(R.R8)),
        ret_ins)))
    alt_accept.append(("xsavec_then_xrstor_same_buffer", (
        mov_eax_7,
        xor_edx,
        Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
        Instruction.create_mem(Code.XSAVEC64_MEM, MemoryOperand(R.R8)),
        Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.R8)),
        Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8)),
        ret_ins)))
    # Default standard-only XRSTOR defines MXCSR when RFBM selects SSE or AVX;
    # opt-in --compacted-xrstor requires RFBM[1] (SSE) because compacted RFBM=4 leaves MXCSR unchanged.
    alt_accept.append(("xrstor_avx_vzeroupper_xsaveopt64_avx", (
        mov_eax_4,
        xor_edx,
        Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
        Instruction.create(Code.VEX_VZEROUPPER),
        Instruction.create_mem(Code.XSAVEOPT64_MEM, MemoryOperand(R.R8)),
        ret_ins)))
    alt_accept.append(("xrstor_sse_vzeroupper_xsaveopt64_avx", (
        mov_eax_2,
        xor_edx,
        Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
        Instruction.create(Code.VEX_VZEROUPPER),
        mov_eax_4,
        Instruction.create_mem(Code.XSAVEOPT64_MEM, MemoryOperand(R.R8)),
        ret_ins)))
    alt_accept.append(("xrstor_x87_xsaveopt64_x87", (
        mov_eax_1,
        xor_edx,
        Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
        Instruction.create_mem(Code.XSAVEOPT64_MEM, MemoryOperand(R.R8)),
        ret_ins)))
    alt_expected = {
        "xrstor_r9_saved_rdx": "op_xrstor(state, r9, rax, rdx, true, false);",
        "xrstor3_vzeroupper_xsave7": "op_vzeroupper(state);",
        "xrstor_xsavec64": "op_xsavec(state, r8, rax, rdx, true);",
        "xsavec_then_xrstor_same_buffer": "op_xrstor(state, r8, rax, rdx, true, false);",
        "xrstor_xsaveopt64": "op_xsaveopt(state, r8, rax, rdx, true);",
        "xrstor_avx_vzeroupper_xsaveopt64_avx": "op_xsaveopt(state, r8, rax, rdx, true);",
        "xrstor_sse_vzeroupper_xsaveopt64_avx": "op_xsaveopt(state, r8, rax, rdx, true);",
        "xrstor_x87_xsaveopt64_x87": "op_xsaveopt(state, r8, rax, rdx, true);",
        "vmovdqa_ymm_roundtrip": "load256(rcx, true)",
        "movdqa_xmm_roundtrip": "store128(r8, get_xmm(state, 3), true);",
        "movq_mm_gpr_roundtrip": "op_emms(state);",
    }
    for label, seq in alt_accept:
        raw = _encode_seq(*seq)
        decoded = list(Decoder(64, raw, ip=0x1000))
        assert len(decoded) == len(seq) and not any(ins.is_invalid for ins in decoded), label
        src, count = lift_code(label, 0x1000, raw)
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if alt_expected[label] not in src:
            raise AssertionError(f"{label}: expected {alt_expected[label]!r} in generated C++")

    canon_raw = _encode_seq(mov_eax_7, xor_edx, Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8)), ret_ins)
    canon_src, canon_count = lift_code("canon_bare", 0x1000, canon_raw, entry_state="canonical")
    assert canon_count == 4 and "Entry-state profile: canonical" in canon_src
    for label, seq in (
        ("xrstor_avx_xsaveopt64_avx", (
            mov_eax_4, xor_edx, Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
            Instruction.create_mem(Code.XSAVEOPT64_MEM, MemoryOperand(R.R8)), ret_ins)),
        ("xrstor_avx_vzeroupper_xsaveopt64_avx", (
            mov_eax_4, xor_edx, Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
            Instruction.create(Code.VEX_VZEROUPPER),
            Instruction.create_mem(Code.XSAVEOPT64_MEM, MemoryOperand(R.R8)), ret_ins)),
    ):
        raw = _encode_seq(*seq)
        src_def, count_def = lift_code(label, 0x1000, raw)
        assert count_def == len(seq) and "Entry-state profile: defined" in src_def, label
        try:
            lift_code(label, 0x1000, raw, compacted_xrstor=True)
        except ValueError as exc:
            assert "stores MXCSR before any restore defined it" in str(exc), exc
        else:
            raise AssertionError(f"{label}: compacted_xrstor=True unexpectedly accepted under defined profile")
        src, count = lift_code(label, 0x1000, raw, entry_state="canonical", compacted_xrstor=True)
        assert count == len(seq) and "Entry-state profile: canonical" in src, label

    # Also verify runtime #GP for bounded mask contract (EDX != 0 / EAX & ~7 != 0)
    # and acceptance of invalid MXCSR when RFBM == 1 (RFBM & 6 == 0).
    seg_xsave = Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8))
    seg_xsave.segment_prefix = R.GS

    unit_cases = [
        # Entry-state & partial-initialization rejections under default 'defined' profile
        ("bare_xsave_default",
         _encode_seq(mov_eax_7, xor_edx, Instruction.create_mem(Code.XSAVE_MEM, MemoryOperand(R.R8)), ret_ins),
         "requires explicit --entry-state canonical"),
        ("bare_xsave64_default",
         _encode_seq(mov_eax_7, xor_edx, Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8)), ret_ins),
         "requires explicit --entry-state canonical"),
        ("partial_sse_to_all_xsave64",
         _encode_seq(mov_eax_2, xor_edx, Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
                     mov_eax_7, Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8)), ret_ins),
         "uninitialized XSAVE component(s) (x87, avx)"),
        ("partial_x87_to_x87_sse_xsave64",
         _encode_seq(mov_eax_1, xor_edx, Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
                     mov_eax_3, Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8)), ret_ins),
         "uninitialized XSAVE component(s) (sse)"),
        ("partial_x87_sse_to_all_xsave64",
         _encode_seq(mov_eax_3, xor_edx, Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
                     mov_eax_7, Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8)), ret_ins),
         "uninitialized XSAVE component(s) (avx)"),
        ("vmovdqa_ymm0_then_xsave64",
         _encode_seq(Instruction.create_reg_mem(Code.VEX_VMOVDQA_YMM_YMMM256, R.YMM0, MemoryOperand(R.RCX)),
                     mov_eax_7, xor_edx, Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8)), ret_ins),
         "requires explicit --entry-state canonical"),
        ("dynamic_mask_after_partial_xrstor",
         _encode_seq(mov_eax_2, xor_edx, Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
                     Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RAX, MemoryOperand(R.RCX)),
                     Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8)), ret_ins),
         "requires explicit --entry-state canonical"),
        # Out-of-scope XSAVE-family & FXSAVE/FXRSTOR forms
        ("bare_xsaveopt_default",
         _encode_seq(mov_eax_7, xor_edx, Instruction.create_mem(Code.XSAVEOPT_MEM, MemoryOperand(R.R8)), ret_ins),
         "requires explicit --entry-state canonical"),
        ("xsaveopt_outside_r8",
         _encode_seq(mov_eax_7, xor_edx, Instruction.create_mem(Code.XSAVEOPT64_MEM, MemoryOperand(R.RCX)), ret_ins),
         "non-r8 store memory operand"),
        ("vzeroupper_xsave64_avx_mxcsr_undefined",
         _encode_seq(Instruction.create(Code.VEX_VZEROUPPER), mov_eax_4, xor_edx,
                     Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8)), ret_ins),
         "stores MXCSR before any restore defined it"),
        ("vzeroupper_xsaveopt64_avx_mxcsr_undefined",
         _encode_seq(Instruction.create(Code.VEX_VZEROUPPER), mov_eax_4, xor_edx,
                     Instruction.create_mem(Code.XSAVEOPT64_MEM, MemoryOperand(R.R8)), ret_ins),
         "stores MXCSR before any restore defined it"),
        ("x87_restore_vzeroupper_xsaveopt64_x87_avx_mxcsr_undefined",
         _encode_seq(mov_eax_1, xor_edx, Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
                     Instruction.create(Code.VEX_VZEROUPPER),
                     Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.EAX, 5),
                     Instruction.create_mem(Code.XSAVEOPT64_MEM, MemoryOperand(R.R8)), ret_ins),
         "MXCSR"),
        ("bare_xsavec_default",
         _encode_seq(mov_eax_7, xor_edx, Instruction.create_mem(Code.XSAVEC64_MEM, MemoryOperand(R.R8)), ret_ins),
         "requires explicit --entry-state canonical"),
        ("xsavec_outside_r8",
         _encode_seq(mov_eax_7, xor_edx, Instruction.create_mem(Code.XSAVEC_MEM, MemoryOperand(R.RCX)), ret_ins),
         "non-r8 store memory operand"),
        ("xsaves", _encode_seq(mov_eax_7, xor_edx, Instruction.create_mem(Code.XSAVES_MEM, MemoryOperand(R.R8)), ret_ins),
         "unsupported XSAVE-family"),
        ("xrstors64", _encode_seq(mov_eax_7, xor_edx, Instruction.create_mem(Code.XRSTORS64_MEM, MemoryOperand(R.RCX)), ret_ins),
         "unsupported XSAVE-family"),
        ("fxsave", _encode_seq(Instruction.create_mem(Code.FXSAVE_M512BYTE, MemoryOperand(R.R8)), ret_ins),
         "unsupported FXSAVE/FXRSTOR"),
        ("fxrstor64", _encode_seq(Instruction.create_mem(Code.FXRSTOR64_M512BYTE, MemoryOperand(R.RCX)), ret_ins),
         "unsupported FXSAVE/FXRSTOR"),
        # EVEX and unsupported vector/GPR registers
        ("evex_vmovdqu64", bytes.fromhex("62f1fe486f01") + _encode_seq(ret_ins), "unsupported EVEX instruction"),
        ("nonvol_xmm6", _encode_seq(
            Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM6, MemoryOperand(R.RCX)), ret_ins),
         "nonvolatile vector register"),
        ("nonvol_ymm6", _encode_seq(
            Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM6, MemoryOperand(R.RCX)), ret_ins),
         "nonvolatile vector register"),
        ("nonvol_zmm0", bytes.fromhex("62f1fe486f01") + _encode_seq(ret_ins), "unsupported EVEX instruction"),
        ("nonvol_rbx", _encode_seq(Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RBX, R.RCX), ret_ins),
         "nonvolatile GPR"),
        ("nonvol_rdi", _encode_seq(Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RDI, R.RCX), ret_ins),
         "nonvolatile GPR"),
        ("rsp_base", _encode_seq(mov_eax_7, xor_edx, Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RSP)), ret_ins),
         "unsupported memory operand"),
        # Unclosed MMX, undefined masks/registers, clobbered/non-argument/non-r8 bases
        ("unclosed_mmx", _encode_seq(
            Instruction.create_reg_mem(Code.MOVQ_MM_MMM64, R.MM0, MemoryOperand(R.RCX)), ret_ins),
         "unclosed MMX state"),
        ("unclosed_mmx_read_gpr", _encode_seq(
            mov_eax_7, xor_edx, Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
            Instruction.create_reg_reg(Code.MOVQ_RM64_MM, R.R9, R.MM3), ret_ins),
         "unclosed MMX state"),
        ("unclosed_mmx_read_mem", _encode_seq(
            mov_eax_7, xor_edx, Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)),
            Instruction.create_mem_reg(Code.MOVQ_MMM64_MM, MemoryOperand(R.R8), R.MM3), ret_ins),
         "unclosed MMX state"),
        ("undef_eax_xsave", _encode_seq(
            xor_edx, Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8)), ret_ins),
         "undefined mask GPR rax"),
        ("undef_ymm1", _encode_seq(
            Instruction.create_mem_reg(Code.VEX_VMOVDQU_YMMM256_YMM, MemoryOperand(R.R8), R.YMM1), ret_ins),
         "undefined vector register ymm1"),
        ("clobbered_rdx_xrstor", _encode_seq(
            mov_eax_7, xor_edx, Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RDX)), ret_ins),
         "clobbered pointer rdx"),
        ("non_arg_base_r9", _encode_seq(
            mov_eax_7, xor_edx, Instruction.create_reg_i32(Code.MOV_R32_IMM32, R.R9D, 0),
            Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.R9)), ret_ins),
         "unsupported memory base"),
        ("store_outside_r8", _encode_seq(
            mov_eax_7, xor_edx, Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.RCX)), ret_ins),
         "non-r8 store memory operand"),
        ("store_via_saved_rcx", _encode_seq(
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R9, R.RCX),
            mov_eax_7, xor_edx, Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R9)), ret_ins),
         "non-r8 store memory operand"),
        ("write_r8", _encode_seq(Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R8, R.RCX), ret_ins),
         "unsupported write to output buffer pointer r8"),
        # Prefixes, addressing, control flow, stream boundaries
        ("segment_prefix", _encode_seq(mov_eax_7, xor_edx, seg_xsave, ret_ins), "unsupported instruction prefix"),
        ("lock_prefix", bytes.fromhex("f00fb109") + _encode_seq(ret_ins), "unsupported instruction prefix"),
        ("rip_rel_mem", _encode_seq(
            mov_eax_7, xor_edx,
            Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RIP, displ=0x1020, displ_size=1)), ret_ins),
         "unsupported memory operand"),
        ("indexed_mem", _encode_seq(
            mov_eax_7, xor_edx,
            Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX, R.R9, 1)), ret_ins),
         "unsupported memory operand"),
        ("branch_jmp", _encode_seq(Instruction.create_branch(Code.JMP_REL8_64, 0x1010), ret_ins),
         "unsupported instruction"),
        ("missing_ret", _encode_seq(mov_eax_7, xor_edx), "missing terminal ret"),
        ("trailing_after_ret", _encode_seq(mov_eax_7, ret_ins, xor_edx), "unreachable instruction after ret"),
        ("empty_stream", b"", "truncated instruction stream"),
        ("invalid_bytes", bytes.fromhex("fffe") + _encode_seq(ret_ins), "invalid instruction"),
    ]

    for label, raw, expected_reason in unit_cases:
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if expected_reason not in str(exc):
                raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator unexpectedly accepted rejected control")

    # End-to-end PE patching controls on a copy of the built native DLL.
    pe_bytes = original_path.read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_xrstor64_xsave64")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(
            i for i in range(0, len(records), 12)
            if struct.unpack_from("<I", records, i)[0] == entry
        )
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        orig_end = struct.unpack_from("<I", records, rec_idx + 4)[0]
        max_len = orig_end - entry

    unit_map = {lbl: raw for lbl, raw, _ in unit_cases}
    pe_patches = [
        ("pe_xsaveopt_outside_r8", unit_map["xsaveopt_outside_r8"], "non-r8 store memory operand"),
        ("pe_xsavec_outside_r8", unit_map["xsavec_outside_r8"], "non-r8 store memory operand"),
        ("pe_fxsave", unit_map["fxsave"], "unsupported FXSAVE/FXRSTOR"),
        ("pe_evex", unit_map["evex_vmovdqu64"], "unsupported EVEX instruction"),
        ("pe_nonvol_ymm6", unit_map["nonvol_ymm6"], "nonvolatile vector register"),
        ("pe_unclosed_mmx", unit_map["unclosed_mmx"], "unclosed MMX state"),
        ("pe_store_outside_r8", unit_map["store_outside_r8"], "non-r8 store memory operand"),
        ("pe_clobbered_rdx", unit_map["clobbered_rdx_xrstor"], "clobbered pointer rdx"),
        ("pe_trailing_after_ret", unit_map["trailing_after_ret"], "unreachable instruction after ret"),
        ("pe_invalid_bytes", unit_map["invalid_bytes"], "invalid instruction"),
    ]
    with TemporaryDirectory() as folder:
        patched_path = Path(folder) / "patched.dll"
        for label, patch, expected_reason in pe_patches:
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch ({len(patch)} bytes) exceeds op_xrstor64_xsave64 ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off : file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched_path.write_bytes(damaged)
            try:
                translate(patched_path, ["op_xrstor64_xsave64"])
            except ValueError as exc:
                if expected_reason not in str(exc):
                    raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"{label}: PE translate accepted rejected patch")

    for path, export_name, compacted_opt in (
        (original_path, "op_xsave_only", False),
        (original_path, "op_xsave64_only", False),
        (original_path, "op_xrstor64_avx_xsaveopt64_avx", True),
        (altered_path, "op_xrstor64_mask_sse", False),
    ):
        try:
            translate(path, [export_name], compacted_xrstor=compacted_opt)
        except ValueError as exc:
            if "requires explicit --entry-state canonical" not in str(exc):
                raise AssertionError(f"{export_name}: unexpected error {exc!r}") from exc
        else:
            raise AssertionError(f"{export_name}: default profile unexpectedly accepted uninitialized entry save")

    def_avx_src, _ = translate(original_path, ["op_xrstor64_avx_xsaveopt64_avx"])
    assert "Entry-state profile: defined" in def_avx_src
    canon_pe_src, _ = translate(original_path, list(CANONICAL_EXPORTS), entry_state="canonical", compacted_xrstor=True)
    assert "Entry-state profile: canonical" in canon_pe_src
    canon_alt_src, _ = translate(altered_path, ["op_xrstor64_mask_sse"], entry_state="canonical")
    assert "Entry-state profile: canonical" in canon_alt_src

    print(
        f"{len(alt_accept)} alternate-encoding lift checks, {len(unit_cases)} unit rejection controls, "
        f"and {len(pe_patches)} PE-patched rejection controls passed"
    )


if __name__ == "__main__":
    orig, lifted, alt, canonical, canonical_alt, native_alt, compacted, compacted_alt = (
        Path(p) for p in sys.argv[1:9]
    )
    check_alt_encodings_and_rejections(orig, native_alt)
    check_differential(orig, lifted, alt, canonical, canonical_alt, native_alt, compacted, compacted_alt)
