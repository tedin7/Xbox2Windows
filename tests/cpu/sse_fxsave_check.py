"""Differential checks, alignment/MXCSR fault controls, dead-read ordering, mutation controls and rejection controls for SSE FXSAVE/FXRSTOR AOT."""

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
from cpu_sse_fxsave_aot import lift_code, translate

GUARD_PRE = bytes(0xA0 + (i % 16) for i in range(64))
GUARD_POST = bytes(0xC0 + (i % 16) for i in range(64))
BUF_LEN = 640
FILL = b"\x5A" * BUF_LEN

CANONICAL_EXPORTS = (
    "op_fxsave_only",
    "op_fxsave64_only",
)

DEFINED_EXPORTS = (
    "op_fxrstor_fxsave",
    "op_fxrstor64_fxsave64",
    "op_fxrstor_fxsave64",
    "op_fxrstor64_fxsave",
    "op_fxrstor64_disp",
    "op_pipeline",
    "op_fxrstor64_movq_store",
    "op_fxrstor64_movq_gpr",
)

EXPORTS = CANONICAL_EXPORTS + DEFINED_EXPORTS

EXPORT_MODES = {
    "op_fxsave_only": (True, False, 0),
    "op_fxsave64_only": (True, True, 0),
    "op_fxrstor_fxsave": (False, False, 0),
    "op_fxrstor64_fxsave64": (True, True, 0),
    "op_fxrstor_fxsave64": (False, True, 0),
    "op_fxrstor64_fxsave": (True, False, 0),
    "op_fxrstor64_disp": (True, True, 64),
    "op_pipeline": (True, True, 0),
    "op_fxrstor64_movq_store": (True, True, 0),
    "op_fxrstor64_movq_gpr": (True, True, 0),
}

ALTERED_EXPORTS = (
    "op_fxrstor64_fxsave64",
    "op_fxrstor_fxsave64",
    "op_fxrstor64_disp",
    "op_pipeline",
)


# --- Native FXSAVE/FXRSTOR caller-state trampoline ---

def _encode_seq(*instructions) -> bytes:
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


class NativeHarness:
    """VirtualAlloc x64 caller harness preserving caller XMM6..15, x87/MMX and MXCSR."""

    def __init__(self):
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
        k32.VirtualAlloc.restype = ctypes.c_void_p
        k32.VirtualFree.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
        k32.VirtualFree.restype = wintypes.BOOL
        k32.RtlAddFunctionTable.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.c_uint64]
        k32.RtlAddFunctionTable.restype = ctypes.c_ubyte
        k32.RtlDeleteFunctionTable.argtypes = [ctypes.c_void_p]
        k32.RtlDeleteFunctionTable.restype = ctypes.c_ubyte
        self._k32 = k32

        # fn_save(rcx=buf) -> fxsave64 [rcx]; ret
        save_code = _encode_seq(
            Instruction.create_mem(Code.FXSAVE64_M512BYTE, MemoryOperand(R.RCX)),
            Instruction.create(Code.RETNQ),
        )
        # fn_rstor(rcx=buf) -> fxrstor64 [rcx]; ret
        rstor_code = _encode_seq(
            Instruction.create_mem(Code.FXRSTOR64_M512BYTE, MemoryOperand(R.RCX)),
            Instruction.create(Code.RETNQ),
        )
        # 32-bit variants for independent host FIP/FDP capability probing.
        save32_code = _encode_seq(
            Instruction.create_mem(Code.FXSAVE_M512BYTE, MemoryOperand(R.RCX)),
            Instruction.create(Code.RETNQ),
        )
        rstor32_code = _encode_seq(
            Instruction.create_mem(Code.FXRSTOR_M512BYTE, MemoryOperand(R.RCX)),
            Instruction.create(Code.RETNQ),
        )
        # fn_call(rcx=target_fn, rdx=args_block) with SEH unwind metadata:
        #   prolog: push rbx (1B); sub rsp, 32 (4B) -> 5B total
        call_code = _encode_seq(
            Instruction.create_reg(Code.PUSH_R64, R.RBX),
            Instruction.create_reg_i32(Code.SUB_RM64_IMM8, R.RSP, 32),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R10, R.RCX),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RBX, R.RDX),
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RAX, MemoryOperand(R.RBX, displ=24, displ_size=1)),
            Instruction.create_mem(Code.FXSAVE64_M512BYTE, MemoryOperand(R.RAX)),
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RAX, MemoryOperand(R.RBX, displ=32, displ_size=1)),
            Instruction.create_mem(Code.FXRSTOR64_M512BYTE, MemoryOperand(R.RAX)),
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RCX, MemoryOperand(R.RBX, displ=0)),
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RDX, MemoryOperand(R.RBX, displ=8, displ_size=1)),
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.R8, MemoryOperand(R.RBX, displ=16, displ_size=1)),
            Instruction.create_reg(Code.CALL_RM64, R.R10),
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RAX, MemoryOperand(R.RBX, displ=24, displ_size=1)),
            Instruction.create_mem(Code.FXRSTOR64_M512BYTE, MemoryOperand(R.RAX)),
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

        self._save_buf, self._save_ptr, _, _, _ = alloc_aligned_buffer(b"\x00" * 512)
        self._clean_buf, self._clean_ptr, _, _, _ = alloc_aligned_buffer(clean_entry_bytes())
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

    def probe_fx_err_ptrs(self, mxcsr_mask: int) -> dict[tuple[bool, bool, bool], tuple[bool, bool, bool, bool, bool]]:
        """Observe whether native FXRSTOR* -> FXSAVE* preserves FOP/FIP/FDP and FCS/FDS for (rstor64, save64, FSW.ES)."""
        caps: dict[tuple[bool, bool, bool], tuple[bool, bool, bool, bool, bool]] = {}
        seed_fop = 0x0357
        seed_fip = 0x00007FFF12345678
        seed_fdp = 0x00007FFF87654321
        exp_fop = struct.pack("<H", seed_fop)
        seeds_cs_ds = (0x3B, 0x2B)
        for rstor64 in (False, True):
            eff_fip = seed_fip if rstor64 else (seed_fip & 0xFFFFFFFF)
            eff_fdp = seed_fdp if rstor64 else (seed_fdp & 0xFFFFFFFF)
            for save64 in (False, True):
                width = 8 if save64 else 4
                exp_fip = struct.pack("<Q", eff_fip)[:width]
                exp_fdp = struct.pack("<Q", eff_fdp)[:width]
                for es in (False, True):
                    img = make_fx_image(
                        mxcsr_mask,
                        fsw=0x0080 if es else 0x0000,
                        fop=seed_fop,
                        fip=seed_fip,
                        fdp=seed_fdp,
                        fcs=0 if rstor64 else seeds_cs_ds[0],
                        fds=0 if rstor64 else seeds_cs_ds[1],
                        mxcsr=0x1F80,
                        seed=1,
                        poison_reserved=False,
                    )
                    in_buf, in_ptr, _, _, _ = alloc_aligned_buffer(img)
                    out_buf, out_ptr, _, out_off, out_post = alloc_aligned_buffer(b"\x5A" * 512)
                    self.save_fn(self._save_ptr)
                    try:
                        (self.rstor_fn if rstor64 else self.rstor32_fn)(in_ptr)
                        (self.save_fn if save64 else self.save32_fn)(out_ptr)
                    finally:
                        self.rstor_fn(self._save_ptr)
                    raw = out_buf.raw[out_off:out_post]
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
                        for i, offset in enumerate((12, 20)):
                            observed = struct.unpack_from("<H", raw, offset)[0]
                            expected = 0 if rstor64 else seeds_cs_ds[i]
                            if observed not in (0, expected):
                                raise AssertionError(f"unexpected host selector at {offset}: {observed:#x}")
                            selectors[i] = not rstor64 and observed == expected
                    caps[(rstor64, save64, es)] = (
                        got_fop == exp_fop, got_fip == exp_fip, got_fdp == exp_fdp, *selectors)
        return caps


# --- Independent Python FXSAVE/FXRSTOR reference model ---

def clean_entry_bytes(mxcsr_mask: int = 0xFFFF) -> bytes:
    buf = bytearray(512)
    struct.pack_into("<HHBBH", buf, 0, 0x037F, 0, 0, 0, 0)
    struct.pack_into("<II", buf, 24, 0x1F80, mxcsr_mask)
    return bytes(buf)


class FxModel:
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

    def fxrstor(self, buf: bytes, is64: bool) -> None:
        new_mxcsr = struct.unpack_from("<I", buf, 24)[0]
        if new_mxcsr & ~self.mxcsr_mask:
            raise ValueError(f"reserved MXCSR bits set: {new_mxcsr:#x}")
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
        self.mxcsr = new_mxcsr
        top = (self.fsw >> 11) & 7
        for i in range(8):
            phys = (top + i) & 7
            self.st[phys][0:10] = buf[32 + i * 16 : 32 + i * 16 + 10]
            self.st[phys][10:16] = b"\x00" * 6
        for i in range(16):
            self.xmm[i][:] = buf[160 + i * 16 : 160 + i * 16 + 16]

    def fxsave(self, dst: bytearray, offset: int, is64: bool) -> None:
        out = bytearray(416)
        struct.pack_into("<HHBBH", out, 0, self.fcw, self.fsw & 0x7F7F, self.ftw, 0, self.fop & 0x07FF)
        if is64:
            struct.pack_into("<QQ", out, 8, self.fip, self.fdp)
        else:
            fcs_out = 0 if self.deprecates_fpu_cs_ds else self.fcs
            fds_out = 0 if self.deprecates_fpu_cs_ds else self.fds
            struct.pack_into("<IHH", out, 8, self.fip & 0xFFFFFFFF, fcs_out, 0)
            struct.pack_into("<IHH", out, 16, self.fdp & 0xFFFFFFFF, fds_out, 0)
        struct.pack_into("<II", out, 24, self.mxcsr, self.mxcsr_mask)
        top = (self.fsw >> 11) & 7
        for i in range(8):
            phys = (top + i) & 7
            out[32 + i * 16 : 32 + i * 16 + 10] = self.st[phys][0:10]
        for i in range(16):
            out[160 + i * 16 : 160 + i * 16 + 16] = self.xmm[i]
        dst[offset : offset + 416] = out

    def enter_mmx(self) -> None:
        """Every MMX instruction but EMMS, one that only reads an MM register included, clears TOP and tags all valid."""
        self.ftw = 0xFF
        self.fsw &= ~0x3800

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
    m = FxModel(mxcsr_mask, deprecates_cs_ds)
    if name == "op_fxsave_only":
        m.fxsave(out, 0, is64=False)
    elif name == "op_fxsave64_only":
        m.fxsave(out, 0, is64=True)
    elif name == "op_fxrstor_fxsave":
        m.fxrstor(a, is64=False)
        m.fxsave(out, 0, is64=False)
    elif name == "op_fxrstor64_fxsave64":
        m.fxrstor(a, is64=True)
        m.fxsave(out, 0, is64=not altered)
    elif name == "op_fxrstor_fxsave64":
        m.fxrstor(a, is64=altered)
        m.fxsave(out, 0, is64=True)
    elif name == "op_fxrstor64_fxsave":
        m.fxrstor(a, is64=True)
        m.fxsave(out, 0, is64=False)
    elif name == "op_fxrstor64_disp":
        m.fxrstor(a[64:], is64=not altered)
        m.fxsave(out, 64, is64=True)
    elif name in ("op_fxrstor64_movq_store", "op_fxrstor64_movq_gpr"):
        m.fxrstor(a, is64=True)
        mm3 = m.get_mm(3)
        if name == "op_fxrstor64_movq_store":
            struct.pack_into("<Q", out, 512, mm3)
        m.fxsave(out, 0, is64=True)
    elif name == "op_pipeline":
        m.fxrstor(a, is64=True)
        mm0 = struct.unpack_from("<Q", b, 0)[0]
        m.set_mm(0, mm0)
        rax = m.get_mm(0)
        m.set_mm(7, rax)
        xmm0 = ref_paddd(b[16:32], b[32:48])
        xmm1 = ref_paddd(xmm0, bytes(m.xmm[2]))
        out[512:528] = xmm1
        m.xmm[0][:] = xmm0
        m.xmm[1][:] = xmm1
        m.fxsave(out, 0, is64=not altered)
    else:
        raise ValueError(f"unknown export {name}")
    return bytes(out)


# --- Test vectors ---

def make_fx_image(
    mxcsr_mask: int,
    *,
    fcw: int = 0x037F,
    fsw: int = 0,
    ftw: int = 0,
    fop: int = 0,
    fip: int = 0,
    fcs: int = 0,
    fdp: int = 0,
    fds: int = 0,
    mxcsr: int = 0x1F80,
    seed: int = 1,
    poison_reserved: bool = True,
) -> bytes:
    buf = bytearray(b"\xAA" * BUF_LEN if poison_reserved else b"\x00" * BUF_LEN)
    struct.pack_into("<HHBBH", buf, 0, fcw, fsw, ftw, 0xBB if poison_reserved else 0, fop)
    # Place canonical 64-bit FIP/FDP; for 32-bit tests fcs/fds sit at +12/+20
    # while keeping +14/+22 zero so 64-bit fxrstor64 sees canonical pointers.
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
    return bytes(buf)


def fx_pairs(mxcsr_mask: int):
    rng = random.Random(0xF5A7E01)
    # Canonical 48-bit-sign-extended 64-bit FIP/FDP values avoid hardware-specific
    # non-canonical FIP truncation differences between CPU vendors.
    states = [
        make_fx_image(mxcsr_mask, fcw=0x037F, fsw=0x0000, ftw=0x00, fop=0x0000,
                      fip=0x0000000000000000, fdp=0x0000000000000000, mxcsr=0x1F80, seed=10),
        make_fx_image(mxcsr_mask, fcw=0x027F, fsw=0xFFFF, ftw=0xA5, fop=0xFFFF,
                      fip=0x00007FFF12345678, fdp=0xFFFF800087654321, mxcsr=0x1F80, seed=11),
        make_fx_image(mxcsr_mask, fcw=0x0000, fsw=0x8080, ftw=0xFF, fop=0xF800,
                      fip=0x0000123456789ABC, fdp=0x0000FEDCBA987654, mxcsr=0x3F80, seed=12),
        make_fx_image(mxcsr_mask, fcw=0x037F, fsw=0x7F7F, ftw=0x55, fop=0x07FF,
                      fip=0x00000000ABCDEF01, fcs=0x1234, fdp=0x0000000013572468, fds=0x5678,
                      mxcsr=0x5FBF, seed=13),
        make_fx_image(mxcsr_mask, fcw=0xFFFF, fsw=0x3800, ftw=0xAA, fop=0x1234,
                      fip=0xFFFF800011223344, fdp=0x00007FFF55667788, mxcsr=0x9FC0, seed=14),
    ]
    pairs = []
    for idx, a in enumerate(states):
        b = rng.randbytes(BUF_LEN)
        if idx == 4:
            # Make b carry valid 64-byte displaced image too for disp checks.
            b = b"\xCC" * 64 + a[: BUF_LEN - 64]
        pairs.append((a, b))
    return pairs


def disp_pairs(mxcsr_mask: int):
    base_pairs = fx_pairs(mxcsr_mask)
    return [(b"\xEE" * 64 + a[: BUF_LEN - 64], b) for a, b in base_pairs]


# --- 64-byte guarded buffer helpers ---

def alloc_aligned_buffer(payload: bytes, align: int = 16, offset: int = 0):
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
    """Adapt native FOP/FIP/FDP and selector expectations to independent measurements."""
    rstor64, save64, off = EXPORT_MODES[name]
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


def assert_equal_output(label: str, name: str, idx, got: bytes, want: bytes) -> None:
    if got != want:
        diff = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)
        raise AssertionError(
            f"{label} {name} case {idx}: first mismatch at byte {diff}: "
            f"got {got[diff : diff + 16].hex()} != want {want[diff : diff + 16].hex()}"
        )


def bind_exports(dll_path: Path, prefix: str = "", names=EXPORTS):
    lib = ctypes.CDLL(str(dll_path.resolve()))
    funcs = {}
    for name in names:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = None
        funcs[name] = fn
    return lib, funcs


# --- Negative runtime controls: alignment (+8), invalid MXCSR (0x11F80), dead-read, canary ---

def check_alignment_and_mxcsr_faults(
    harness: NativeHarness, native_fns, lifted_fns, canonical_fns, mxcsr_mask: int
) -> tuple[int, int]:
    valid_a = make_fx_image(mxcsr_mask, fip=0x00007FFF11223344, fdp=0x00007FFF55667788, seed=99)
    invalid_mxcsr_a = bytearray(valid_a)
    struct.pack_into("<I", invalid_mxcsr_a, 24, 0x11F80)  # bit 16 (reserved) set

    misalign_checks = 0
    for name in CANONICAL_EXPORTS:
        for label, is_native, fn in (
            ("native", True, native_fns[name]),
            ("canonical-lifted", False, canonical_fns[name]),
        ):
            buf_a, ptr_a, _, _, _ = alloc_aligned_buffer(valid_a)
            buf_out, ptr_out, pre, off, post = alloc_aligned_buffer(FILL, offset=8)
            try:
                if is_native:
                    harness.call_native(fn, ptr_a, ptr_a, ptr_out)
                else:
                    fn(ptr_a, ptr_a, ptr_out)
            except OSError as exc:
                assert "access violation reading 0xffffffffffffffff" in str(exc).lower(), (name, label, exc)
            else:
                raise AssertionError(f"{label} {name} accepted +8 misaligned output")
            assert check_guards_and_read(buf_out, pre, off, post, label) == FILL
            _ = buf_a
            misalign_checks += 1

    for name in ("op_fxrstor_fxsave", "op_fxrstor64_fxsave64"):
        for label, is_native, fn in (
            ("native", True, native_fns[name]),
            ("lifted", False, lifted_fns[name]),
        ):
            buf_a, ptr_a, _, _, _ = alloc_aligned_buffer(valid_a, offset=8)
            buf_out, ptr_out, pre, off, post = alloc_aligned_buffer(FILL)
            try:
                if is_native:
                    harness.call_native(fn, ptr_a, ptr_a, ptr_out)
                else:
                    fn(ptr_a, ptr_a, ptr_out)
            except OSError as exc:
                assert "access violation reading 0xffffffffffffffff" in str(exc).lower(), (name, label, exc)
            else:
                raise AssertionError(f"{label} {name} accepted +8 misaligned input")
            assert check_guards_and_read(buf_out, pre, off, post, label) == FILL
            _ = buf_a
            misalign_checks += 1

    mxcsr_checks = 0
    for name in ("op_fxrstor_fxsave", "op_fxrstor64_fxsave64"):
        for label, is_native, fn in (
            ("native", True, native_fns[name]),
            ("lifted", False, lifted_fns[name]),
        ):
            buf_a, ptr_a, _, _, _ = alloc_aligned_buffer(bytes(invalid_mxcsr_a))
            buf_out, ptr_out, pre, off, post = alloc_aligned_buffer(FILL)
            try:
                if is_native:
                    harness.call_native(fn, ptr_a, ptr_a, ptr_out)
                else:
                    fn(ptr_a, ptr_a, ptr_out)
            except OSError as exc:
                assert "access violation reading 0xffffffffffffffff" in str(exc).lower(), (name, label, exc)
            else:
                raise AssertionError(f"{label} {name} accepted invalid MXCSR 0x11F80")
            assert check_guards_and_read(buf_out, pre, off, post, label) == FILL
            _ = buf_a
            mxcsr_checks += 1

    return misalign_checks, mxcsr_checks


def check_discarded_read(
    harness: NativeHarness,
    native_path: Path,
    lifted_path: Path,
    mxcsr_mask: int,
    deprecates_cs_ds: bool,
    fp_caps: dict,
) -> None:
    payload = make_fx_image(mxcsr_mask, fip=0x00007FFF12345678, fdp=0x00007FFF87654321, seed=77)
    bad_mxcsr = bytearray(payload)
    struct.pack_into("<I", bad_mxcsr, 24, 0x11F80)

    _, n_fns = bind_exports(native_path, "", ("op_discarded_read",))
    _, l_fns = bind_exports(lifted_path, "lifted_", ("op_discarded_read",))

    want_first = reference("op_fxrstor64_fxsave64", payload, payload, mxcsr_mask, deprecates_cs_ds)
    expected_ok = bytearray(want_first)
    expected_ok[512:528] = payload[160:176]
    native_want_first = expected_native_output(want_first, payload, "op_fxrstor64_fxsave64", fp_caps)
    native_expected_ok = expected_native_output(bytes(expected_ok), payload, "op_fxrstor64_fxsave64", fp_caps)

    for label, is_native, fn in (
        ("native", True, n_fns["op_discarded_read"]),
        ("lifted", False, l_fns["op_discarded_read"]),
    ):
        buf_a, ptr_a, _, _, _ = alloc_aligned_buffer(payload)
        buf_bad, ptr_bad, _, _, _ = alloc_aligned_buffer(bytes(bad_mxcsr))

        # Mode 0: readable valid rdx -> both [0..415] and [512..528] written.
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

        # Mode 1: unmapped rdx=0 -> [0..415] written before fault, [512..528] stays FILL.
        out_unmap, ptr_unmap, pre, off, post = alloc_aligned_buffer(FILL)
        try:
            if is_native:
                harness.call_native(fn, ptr_a, 0, ptr_unmap)
            else:
                fn(ptr_a, 0, ptr_unmap)
        except OSError as exc:
            assert "access violation reading" in str(exc).lower(), (label, exc)
        else:
            raise AssertionError(f"{label} op_discarded_read discarded unmapped fault")
        assert_equal_output(
            label,
            "op_discarded_read",
            "unmapped",
            check_guards_and_read(out_unmap, pre, off, post, label),
            native_want_first if is_native else want_first,
        )

        # Mode 2: invalid MXCSR (0x11F80) on rdx -> [0..415] written before #GP, [512..528] stays FILL.
        out_gp, ptr_gp, pre, off, post = alloc_aligned_buffer(FILL)
        try:
            if is_native:
                harness.call_native(fn, ptr_a, ptr_bad, ptr_gp)
            else:
                fn(ptr_a, ptr_bad, ptr_gp)
        except OSError as exc:
            assert "access violation reading 0xffffffffffffffff" in str(exc).lower(), (label, exc)
        else:
            raise AssertionError(f"{label} op_discarded_read accepted invalid MXCSR 0x11F80")
        assert_equal_output(
            label,
            "op_discarded_read",
            "invalid_mxcsr",
            check_guards_and_read(out_gp, pre, off, post, label),
            native_want_first if is_native else want_first,
        )

        _ = buf_a, buf_bad


def check_lifted_preserves_host_state(
    harness: NativeHarness, lifted_fns, mxcsr_mask: int
) -> None:
    """Verify lifted exports leave host XMM6..15, x87/MMX and MXCSR untouched."""
    canary = make_fx_image(
        mxcsr_mask, fcw=0x027F, fsw=0x0000, ftw=0x00, fop=0x0000,
        mxcsr=0x1F80, seed=555, poison_reserved=False,
    )
    payload, ptr_a, _, _, _ = alloc_aligned_buffer(
        make_fx_image(mxcsr_mask, fcw=0x037F, fsw=0x1234, ftw=0xFF, mxcsr=0x3F80 & mxcsr_mask, seed=88)
    )
    b_data, ptr_b, _, _, _ = alloc_aligned_buffer(bytes(range(BUF_LEN & 0xFF)) * 3)
    out_buf, ptr_out, _, _, _ = alloc_aligned_buffer(FILL)
    save_buf, save_ptr, _, _, _ = alloc_aligned_buffer(b"\x00" * 512)
    canary_buf, canary_ptr, _, _, _ = alloc_aligned_buffer(canary)
    after_buf, after_ptr, _, _, _ = alloc_aligned_buffer(b"\x00" * 512)

    harness.save_fn(save_ptr)
    try:
        harness.rstor_fn(canary_ptr)
        harness.save_fn(canary_ptr)  # re-capture exact hardware canonical fields
        lifted_fns["op_pipeline"](ptr_a, ptr_b, ptr_out)
        harness.save_fn(after_ptr)
        before_bytes = bytes((ctypes.c_ubyte * 416).from_address(canary_ptr))
        after_bytes = bytes((ctypes.c_ubyte * 416).from_address(after_ptr))
        # Check MXCSR, FCW, and nonvolatile XMM6..15 [256..416] remain identical.
        if before_bytes[0:2] != after_bytes[0:2]:
            raise AssertionError("lifted op_pipeline mutated host FCW")
        if before_bytes[24:28] != after_bytes[24:28]:
            raise AssertionError("lifted op_pipeline mutated host MXCSR")
        if before_bytes[256:416] != after_bytes[256:416]:
            raise AssertionError("lifted op_pipeline mutated host nonvolatile XMM6..15")
    finally:
        harness.rstor_fn(save_ptr)
    _ = payload, b_data, out_buf, save_buf, canary_buf, after_buf


def check_entry_state_contracts(
    harness: NativeHarness,
    native_fns,
    lifted_lib,
    altered_lib,
    canonical_lib,
    canonical_fns,
    mxcsr_mask: int,
) -> None:
    """Verify explicit entry-state contract metadata and non-default entry divergence."""
    for lib in (lifted_lib, altered_lib, canonical_lib):
        lib.lifted_entry_state_profile.restype = ctypes.c_char_p
        lib.lifted_requires_canonical_entry.restype = ctypes.c_uint32

    assert lifted_lib.lifted_entry_state_profile() == b"defined"
    assert lifted_lib.lifted_requires_canonical_entry() == 0
    assert altered_lib.lifted_entry_state_profile() == b"defined"
    assert altered_lib.lifted_requires_canonical_entry() == 0
    assert canonical_lib.lifted_entry_state_profile() == b"canonical"
    assert canonical_lib.lifted_requires_canonical_entry() == 1

    # Non-default incoming entry state (MXCSR=0x7FA0, XMM6 low dword=0x12345678):
    # native op_fxsave64_only saves both incoming values, while canonical-lifted
    # explicitly documents and emits canonical entry values (0x1F80 and zero).
    nondefault = bytearray(clean_entry_bytes(mxcsr_mask))
    nondefault_mxcsr = 0x7FA0 & mxcsr_mask
    struct.pack_into("<I", nondefault, 24, nondefault_mxcsr)
    struct.pack_into("<I", nondefault, 160 + 6 * 16, 0x12345678)
    entry_buf, entry_ptr, _, _, _ = alloc_aligned_buffer(bytes(nondefault))
    dummy = b"\x00" * BUF_LEN
    got_n = invoke_native(harness, native_fns["op_fxsave64_only"], dummy, dummy, entry_ptr=entry_ptr)
    got_l = invoke_lifted(canonical_fns["op_fxsave64_only"], dummy, dummy)
    _ = entry_buf

    assert struct.unpack_from("<I", got_n, 24)[0] == nondefault_mxcsr
    assert struct.unpack_from("<I", got_n, 256)[0] == 0x12345678
    assert struct.unpack_from("<I", got_l, 24)[0] == 0x1F80
    assert struct.unpack_from("<I", got_l, 256)[0] == 0
    assert got_n != got_l


def check_differential(
    native_path: Path, lifted_path: Path, altered_path: Path, canonical_path: Path
) -> None:
    lifted_lib, lifted = bind_exports(lifted_path, "lifted_", DEFINED_EXPORTS)
    lifted_lib.host_has_fxsr.restype = ctypes.c_uint32
    lifted_lib.host_deprecates_fpu_cs_ds.restype = ctypes.c_uint32
    lifted_lib.host_fp_err_cpuid_info.restype = ctypes.c_uint32
    lifted_lib.host_mxcsr_mask.restype = ctypes.c_uint32
    if not lifted_lib.host_has_fxsr():
        print("host CPUID.01H:EDX.FXSR[24] == 0; skipping native differential check")
        raise SystemExit(77)

    mxcsr_mask = int(lifted_lib.host_mxcsr_mask())
    deprecates_cs_ds = bool(lifted_lib.host_deprecates_fpu_cs_ds())
    fp_cpuid = int(lifted_lib.host_fp_err_cpuid_info())

    _, native = bind_exports(native_path)
    altered_lib, altered = bind_exports(altered_path, "lifted_", DEFINED_EXPORTS)
    canonical_lib, canonical = bind_exports(canonical_path, "lifted_", CANONICAL_EXPORTS)

    harness = NativeHarness()
    try:
        fp_caps = harness.probe_fx_err_ptrs(mxcsr_mask)
        check_entry_state_contracts(
            harness, native, lifted_lib, altered_lib, canonical_lib, canonical, mxcsr_mask
        )
        misalign_checks, mxcsr_checks = check_alignment_and_mxcsr_faults(
            harness, native, lifted, canonical, mxcsr_mask
        )
        check_discarded_read(harness, native_path, lifted_path, mxcsr_mask, deprecates_cs_ds, fp_caps)
        check_lifted_preserves_host_state(harness, lifted, mxcsr_mask)

        total = 0
        for name in EXPORTS:
            pairs = disp_pairs(mxcsr_mask) if name == "op_fxrstor64_disp" else fx_pairs(mxcsr_mask)
            lifted_fn = canonical[name] if name in CANONICAL_EXPORTS else lifted[name]
            for idx, (a, b) in enumerate(pairs):
                want = reference(name, a, b, mxcsr_mask, deprecates_cs_ds)
                want_native = (
                    want
                    if name in CANONICAL_EXPORTS
                    else expected_native_output(want, a, name, fp_caps)
                )
                got_n = invoke_native(harness, native[name], a, b)
                got_l = invoke_lifted(lifted_fn, a, b)
                assert_equal_output("native", name, idx, got_n, want_native)
                assert_equal_output("lifted", name, idx, got_l, want)
                total += 1

        # Negative control: an MM read that leaves TOP and the tag word alone must be caught.
        a, b = fx_pairs(mxcsr_mask)[0]
        want = reference("op_fxrstor64_movq_gpr", a, b, mxcsr_mask, deprecates_cs_ds)
        unchanged = reference("op_fxrstor64_fxsave64", a, b, mxcsr_mask, deprecates_cs_ds)
        if want == unchanged:
            raise AssertionError("the MM read control does not change the saved state")
        try:
            assert_equal_output("lifted", "op_fxrstor64_movq_gpr", 0, unchanged, want)
        except AssertionError:
            pass
        else:
            raise AssertionError("an MM read that leaves TOP and the tags alone was not caught by the comparator")

        diverged = {}
        for name in ALTERED_EXPORTS:
            diffs = 0
            pairs = disp_pairs(mxcsr_mask) if name == "op_fxrstor64_disp" else fx_pairs(mxcsr_mask)
            for idx, (a, b) in enumerate(pairs):
                want_alt = reference(name, a, b, mxcsr_mask, deprecates_cs_ds, altered=True)
                want_orig = reference(name, a, b, mxcsr_mask, deprecates_cs_ds, altered=False)
                got_alt = invoke_lifted(altered[name], a, b)
                assert_equal_output("altered", name, idx, got_alt, want_alt)
                if got_alt != want_orig:
                    diffs += 1
            if not diffs:
                raise AssertionError(f"altered control {name} did not diverge from original")
            diverged[name] = diffs

        # Output mutation negative control.
        sample_a, sample_b = fx_pairs(mxcsr_mask)[1]
        expected = reference("op_fxrstor64_fxsave64", sample_a, sample_b, mxcsr_mask, deprecates_cs_ds)
        mutated = bytearray(expected)
        mutated[172] ^= 0x01
        try:
            assert_equal_output("mutation", "op_fxrstor64_fxsave64", 0, bytes(mutated), expected)
        except AssertionError:
            pass
        else:
            raise AssertionError("mutated output was not caught by comparator")

        es0_64 = fp_caps[(True, True, False)][:3]
        es1_64 = fp_caps[(True, True, True)][:3]
        es0_32_selectors = fp_caps[(False, False, False)][3:]
        es1_32_selectors = fp_caps[(False, False, True)][3:]
        print(
            f"{total} native/lifted/reference FXSAVE/FXRSTOR cases across {len(EXPORTS)} exports matched the separate native and profile references bit-for-bit "
            f"(MXCSR_MASK={mxcsr_mask:#x}, FCS/FDS_deprecated={int(deprecates_cs_ds)}, "
            f"fp_cpuid={fp_cpuid:#x}, fip_fdp_es0={es0_64}, fip_fdp_es1={es1_64}, "
            f"fcs_fds_es0={es0_32_selectors}, fcs_fds_es1={es1_32_selectors}); "
            f"{misalign_checks} +8 alignment #GP controls, {mxcsr_checks} invalid MXCSR (0x11F80) #GP controls, "
            f"ALTER_OP diverged on {diverged}; entry-state contract, non-default entry control, "
            f"dead-read, canary, and output mutation controls passed"
        )
    finally:
        harness.close()


# --- Alternate-encoding acceptance & translator rejection controls ---

def check_alt_encodings_and_rejections(original_path: Path) -> None:
    ret_ins = Instruction.create(Code.RETNQ)
    emms_ins = Instruction.create(Code.EMMS)
    fxsave_r8 = Instruction.create_mem(Code.FXSAVE_M512BYTE, MemoryOperand(R.R8))
    fxsave64_r8 = Instruction.create_mem(Code.FXSAVE64_M512BYTE, MemoryOperand(R.R8))
    fxrstor_rcx = Instruction.create_mem(Code.FXRSTOR_M512BYTE, MemoryOperand(R.RCX))
    fxrstor_rdx = Instruction.create_mem(Code.FXRSTOR_M512BYTE, MemoryOperand(R.RDX))
    fxrstor64_r8 = Instruction.create_mem(Code.FXRSTOR64_M512BYTE, MemoryOperand(R.R8))
    ld_xmm0 = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    st_xmm0 = Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)

    alt_accept = [
        ("fxrstor_rdx_r8", (fxrstor_rdx, fxrstor64_r8, fxsave_r8, ret_ins)),
        ("movdqu_unaligned", (
            Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=1, displ_size=1)),
            Instruction.create_mem_reg(Code.MOVDQU_XMMM128_XMM, MemoryOperand(R.R8, displ=3, displ_size=1), R.XMM0),
            ret_ins)),
        ("movq_r9_r10_bridge", (
            Instruction.create_reg_mem(Code.MOVQ_MM_MMM64, R.MM0, MemoryOperand(R.RCX)),
            Instruction.create_reg_reg(Code.MOVQ_RM64_MM, R.R9, R.MM0),
            Instruction.create_reg_reg(Code.MOVQ_MM_RM64, R.MM3, R.R9),
            Instruction.create_mem_reg(Code.MOVQ_MMM64_MM, MemoryOperand(R.R8), R.MM3),
            emms_ins, ret_ins)),
        ("mov_gpr_r11", (
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.R11, MemoryOperand(R.RCX)),
            Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.R8), R.R11),
            fxrstor_rcx, fxsave64_r8, ret_ins)),
    ]
    alt_expected = {
        "fxrstor_rdx_r8": "op_fxrstor(state, r8, true);",
        "movdqu_unaligned": "store128((r8 + 0x3ull), get_xmm(state, 0), false);",
        "movq_r9_r10_bridge": "r9 = get_mm(state, 0);",
        "mov_gpr_r11": "r11 = load64(rcx);",
    }
    for label, seq in alt_accept:
        raw = _encode_seq(*seq)
        decoded = list(Decoder(64, raw, ip=0x1000))
        assert len(decoded) == len(seq) and not any(ins.is_invalid for ins in decoded), label
        src, count = lift_code(label, 0x1000, raw)
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if alt_expected[label] not in src:
            raise AssertionError(f"{label}: expected {alt_expected[label]!r} in generated C++")

    canonical_raw = _encode_seq(fxsave64_r8, ret_ins)
    canonical_src, canonical_count = lift_code("canonical_bare", 0x1000, canonical_raw, entry_state="canonical")
    assert canonical_count == 2 and "Entry-state profile: canonical" in canonical_src

    evex_vpaddd = Instruction.create_reg_reg_reg(
        Code.EVEX_VPADDD_XMM_K1Z_XMM_XMMM128B32, R.XMM0, R.XMM0, R.XMM1
    )
    evex_vpaddd.op_mask = R.K1
    seg_fxsave = Instruction.create_mem(Code.FXSAVE_M512BYTE, MemoryOperand(R.R8))
    seg_fxsave.segment_prefix = R.GS

    unit_cases = [
        # XSAVE-family rejections
        ("xsave", _encode_seq(Instruction.create_mem(Code.XSAVE_MEM, MemoryOperand(R.R8)), ret_ins), "unsupported XSAVE-family instruction"),
        ("xsave64", _encode_seq(Instruction.create_mem(Code.XSAVE64_MEM, MemoryOperand(R.R8)), ret_ins), "unsupported XSAVE-family instruction"),
        ("xrstor", _encode_seq(Instruction.create_mem(Code.XRSTOR_MEM, MemoryOperand(R.RCX)), ret_ins), "unsupported XSAVE-family instruction"),
        ("xrstor64", _encode_seq(Instruction.create_mem(Code.XRSTOR64_MEM, MemoryOperand(R.RCX)), ret_ins), "unsupported XSAVE-family instruction"),
        ("xsaveopt", _encode_seq(Instruction.create_mem(Code.XSAVEOPT_MEM, MemoryOperand(R.R8)), ret_ins), "unsupported XSAVE-family instruction"),
        ("xsaveopt64", _encode_seq(Instruction.create_mem(Code.XSAVEOPT64_MEM, MemoryOperand(R.R8)), ret_ins), "unsupported XSAVE-family instruction"),
        ("xsavec", _encode_seq(Instruction.create_mem(Code.XSAVEC_MEM, MemoryOperand(R.R8)), ret_ins), "unsupported XSAVE-family instruction"),
        ("xsavec64", _encode_seq(Instruction.create_mem(Code.XSAVEC64_MEM, MemoryOperand(R.R8)), ret_ins), "unsupported XSAVE-family instruction"),
        ("xsaves", _encode_seq(Instruction.create_mem(Code.XSAVES_MEM, MemoryOperand(R.R8)), ret_ins), "unsupported XSAVE-family instruction"),
        ("xrstors", _encode_seq(Instruction.create_mem(Code.XRSTORS_MEM, MemoryOperand(R.RCX)), ret_ins), "unsupported XSAVE-family instruction"),
        # VEX / EVEX rejections
        ("vex_vmovdqa", _encode_seq(Instruction.create_reg_mem(Code.VEX_VMOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)), ret_ins), "unsupported VEX/EVEX instruction"),
        ("vex_vpaddd", _encode_seq(ld_xmm0, Instruction.create_reg_reg_reg(Code.VEX_VPADDD_XMM_XMM_XMMM128, R.XMM0, R.XMM0, R.XMM0), ret_ins), "unsupported VEX/EVEX instruction"),
        ("vex_vzeroupper", _encode_seq(Instruction.create(Code.VEX_VZEROUPPER), ret_ins), "unsupported VEX/EVEX instruction"),
        ("evex_vpaddd", _encode_seq(ld_xmm0, Instruction.create_reg_reg(Code.MOVDQA_XMM_XMMM128, R.XMM1, R.XMM0), evex_vpaddd, ret_ins), "unsupported VEX/EVEX instruction"),
        # maskmovq / maskmovdqu rejections
        ("maskmovq", _encode_seq(Instruction.create_reg_mem(Code.MOVQ_MM_MMM64, R.MM0, MemoryOperand(R.RCX)),
                                 Instruction.create_maskmovq(64, R.MM0, R.MM0), emms_ins, ret_ins), "unsupported masked-store instruction"),
        ("maskmovdqu", _encode_seq(ld_xmm0, Instruction.create_maskmovdqu(64, R.XMM0, R.XMM0), ret_ins), "unsupported masked-store instruction"),
        # Nonvolatile registers
        ("nonvolatile_xmm6", _encode_seq(Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM6, MemoryOperand(R.RCX)), ret_ins), "nonvolatile vector register"),
        ("nonvolatile_xmm15", _encode_seq(fxrstor_rcx, Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8), R.XMM15), ret_ins), "nonvolatile vector register"),
        ("nonvolatile_rbx", _encode_seq(Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RBX, MemoryOperand(R.RCX)), ret_ins), "nonvolatile GPR"),
        ("nonvolatile_rdi", _encode_seq(Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RDI, MemoryOperand(R.RCX)), ret_ins), "nonvolatile GPR"),
        ("nonvolatile_rsp", _encode_seq(Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.RSP), ret_ins), "nonvolatile GPR"),
        # Non-arg bases, non-r8 stores, RIP-relative, indexed, clobbered pointers
        ("non_arg_base_rax", _encode_seq(Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RAX, MemoryOperand(R.RCX)),
                                         Instruction.create_mem(Code.FXRSTOR_M512BYTE, MemoryOperand(R.RAX)), ret_ins), "unsupported memory base"),
        ("nonvolatile_base_rbx", _encode_seq(Instruction.create_mem(Code.FXRSTOR_M512BYTE, MemoryOperand(R.RBX)), ret_ins), "nonvolatile GPR base"),
        ("stack_base_rsp", _encode_seq(Instruction.create_mem(Code.FXSAVE64_M512BYTE, MemoryOperand(R.RSP)), ret_ins), "unsupported memory operand"),
        ("addr32_ecx_base", bytes.fromhex("670fae01") + _encode_seq(ret_ins), "unsupported memory operand"),
        ("store_outside_r8_fxsave", _encode_seq(Instruction.create_mem(Code.FXSAVE_M512BYTE, MemoryOperand(R.RCX)), ret_ins), "non-r8 store memory operand"),
        ("store_outside_r8_fxsave64", _encode_seq(Instruction.create_mem(Code.FXSAVE64_M512BYTE, MemoryOperand(R.RDX)), ret_ins), "non-r8 store memory operand"),
        ("store_outside_r8_movdqa", _encode_seq(ld_xmm0, Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.RCX), R.XMM0), ret_ins), "non-r8 store memory operand"),
        ("write_r8_gpr", _encode_seq(Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R8, R.RCX), ret_ins), "unsupported write to output buffer pointer r8"),
        ("clobbered_rcx_read", _encode_seq(Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RCX, R.RDX), fxrstor_rcx, ret_ins), "memory operand uses clobbered pointer rcx"),
        ("rip_rel_fxsave", _encode_seq(Instruction.create_mem(Code.FXSAVE_M512BYTE, MemoryOperand(R.RIP, displ=0x1040, displ_size=1)), ret_ins), "unsupported memory operand"),
        ("indexed_fxrstor", _encode_seq(Instruction.create_mem(Code.FXRSTOR_M512BYTE, MemoryOperand(R.RCX, R.RDX, 1)), ret_ins), "unsupported memory operand"),
        # Entry-state contract rejections under default 'defined' profile
        ("bare_fxsave_default", _encode_seq(fxsave_r8, ret_ins), "requires explicit --entry-state canonical"),
        ("bare_fxsave64_default", _encode_seq(fxsave64_r8, ret_ins), "requires explicit --entry-state canonical"),
        ("partial_xmm_then_fxsave64", _encode_seq(ld_xmm0, fxsave64_r8, ret_ins), "requires explicit --entry-state canonical"),
        ("partial_emms_then_fxsave", _encode_seq(emms_ins, fxsave_r8, ret_ins), "requires explicit --entry-state canonical"),
        # Prefixes, branches, unclosed MMX, undefined regs, trailing bytes
        ("segment_prefix", _encode_seq(seg_fxsave, ret_ins), "unsupported instruction prefix"),
        ("lock_prefix", bytes.fromhex("f00fb109") + _encode_seq(ret_ins), "unsupported instruction prefix"),
        ("rep_prefix", bytes.fromhex("f3a4") + _encode_seq(ret_ins), "unsupported instruction prefix"),
        ("branch_jmp", bytes.fromhex("eb00") + _encode_seq(ret_ins), "unsupported instruction"),
        ("call_rel", bytes.fromhex("e800000000") + _encode_seq(ret_ins), "unsupported instruction"),
        ("unclosed_mmx", _encode_seq(fxrstor_rcx, Instruction.create_reg_mem(Code.MOVQ_MM_MMM64, R.MM0, MemoryOperand(R.RCX)), fxsave_r8, ret_ins), "unclosed MMX state before ret"),
        ("undef_xmm2", _encode_seq(ld_xmm0, Instruction.create_reg_reg(Code.PADDD_XMM_XMMM128, R.XMM0, R.XMM2), st_xmm0, ret_ins), "undefined vector register xmm2"),
        ("undef_mm1", _encode_seq(Instruction.create_mem_reg(Code.MOVQ_MMM64_MM, MemoryOperand(R.R8), R.MM1), emms_ins, ret_ins), "undefined MMX register mm1"),
        ("undef_rax", _encode_seq(Instruction.create_reg_reg(Code.MOVQ_MM_RM64, R.MM0, R.RAX), emms_ins, ret_ins), "undefined GPR rax"),
        ("missing_ret", _encode_seq(fxrstor_rcx, fxsave_r8), "missing terminal ret"),
        ("trailing_after_ret", _encode_seq(fxrstor_rcx, fxsave_r8, ret_ins, fxsave64_r8), "unreachable instruction after ret"),
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

    pe_bytes = original_path.read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_pipeline")
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
        ("pe_xsave", unit_map["xsave"], "unsupported XSAVE-family instruction"),
        ("pe_xrstor64", unit_map["xrstor64"], "unsupported XSAVE-family instruction"),
        ("pe_vex_vmovdqa", unit_map["vex_vmovdqa"], "unsupported VEX/EVEX instruction"),
        ("pe_evex_vpaddd", unit_map["evex_vpaddd"], "unsupported VEX/EVEX instruction"),
        ("pe_maskmovq", unit_map["maskmovq"], "unsupported masked-store instruction"),
        ("pe_maskmovdqu", unit_map["maskmovdqu"], "unsupported masked-store instruction"),
        ("pe_nonvolatile_xmm15", unit_map["nonvolatile_xmm15"], "nonvolatile vector register"),
        ("pe_nonvolatile_rbx", unit_map["nonvolatile_rbx"], "nonvolatile GPR"),
        ("pe_store_outside_r8", unit_map["store_outside_r8_fxsave64"], "non-r8 store memory operand"),
        ("pe_rip_rel", unit_map["rip_rel_fxsave"], "unsupported memory operand"),
        ("pe_branch", unit_map["branch_jmp"], "unsupported instruction"),
        ("pe_trailing_after_ret", unit_map["trailing_after_ret"], "unreachable instruction after ret"),
    ]
    with TemporaryDirectory() as folder:
        patched_path = Path(folder) / "patched.dll"
        for label, patch, expected_reason in pe_patches:
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch ({len(patch)} bytes) exceeds op_pipeline ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off : file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched_path.write_bytes(damaged)
            try:
                translate(patched_path, ["op_pipeline"])
            except ValueError as exc:
                if expected_reason not in str(exc):
                    raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"{label}: PE translate accepted rejected patch")

    for export_name in CANONICAL_EXPORTS:
        try:
            translate(original_path, [export_name])
        except ValueError as exc:
            if "requires explicit --entry-state canonical" not in str(exc):
                raise AssertionError(f"{export_name}: unexpected error {exc!r}") from exc
        else:
            raise AssertionError(f"{export_name}: default entry-state profile unexpectedly accepted bare fxsave")

    canon_src, _ = translate(original_path, list(CANONICAL_EXPORTS), entry_state="canonical")
    assert "Entry-state profile: canonical" in canon_src

    print(
        f"{len(alt_accept)} alternate-encoding lift checks, {len(unit_cases)} unit rejection controls, "
        f"and {len(pe_patches)} PE-patched rejection controls passed"
    )


if __name__ == "__main__":
    orig, lifted, alt, canonical = (Path(p) for p in sys.argv[1:5])
    check_differential(orig, lifted, alt, canonical)
    check_alt_encodings_and_rejections(orig)
