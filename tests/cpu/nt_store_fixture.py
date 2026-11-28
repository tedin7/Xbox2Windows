#!/usr/bin/env python3
"""Emit the non-temporal store differential fixture (MOVNTPS, MOVNTPD, MOVNTDQ, MOVNTI).

For the same inputs, compare what the host executes for each store against what the shared lowering
(tools/ffx_version_query.lower()) produces for it:

  native  - the exact bytes in an executable section, called directly.
  lifted  - the same instructions lowered to C++ and compiled alongside.

Each case is called as f(target, source, value). The XMM cases load XMM0 or XMM2 from `source` with MOVDQU
and store it to `target`; the MOVNTI cases store the low 32 or 64 bits of `value`. Every case ends with SFENCE
and returns RAX. The native and lifted calls write into two separate 64-byte buffers that start filled with a
sentinel, so the check compares the whole buffer, not only the stored bytes, and an independent reference
(the source bytes or the value at the target offset, the sentinel elsewhere) is checked as well.

The fault cases run under a guard that records the exception code and its two parameters:
  - an XMM store to a target that is not 16-byte aligned raises #GP(0), which Windows reports as an access
    violation at address -1; nothing may be written;
  - a store to a PAGE_NOACCESS page raises an access violation for a write at that address;
  - MOVNTI to an odd address is not a fault.

Controls that must break a rule, so the cases above are informative:
  - `noalign`: the lifted MOVNTPS with its alignment test removed and a plain store in place of the streaming
    one must store where the native one faults (the streaming intrinsic is MOVNTDQ and faults by itself);
  - `nostore`: the native bytes with the store replaced by a NOP must leave the reference unmatched;
  - a plain MOVUPS to the same misaligned target must not fault natively, so the fault above comes from the
    alignment rule and not from the mapping.

usage: nt_store_fixture.py emit <output.cpp>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from iced_x86 import Decoder
from ffx_version_query import INTEGER_HEADER, lower

LOAD_XMM0 = bytes.fromhex("f30f6f02")    # movdqu xmm0, [rdx]
LOAD_XMM2 = bytes.fromhex("f30f6f12")    # movdqu xmm2, [rdx]
VALUE_RAX = bytes.fromhex("4c89c0")      # mov rax, r8
EPILOGUE = bytes.fromhex("0faef8c3")     # sfence ; ret
ALIGN = "if (target & 15u) raise_aligned_m128_fault();"

PAGE = "page"   # the target is the PAGE_NOACCESS page

# (name, description, prologue, store bytes, offset of RCX in the 64-byte buffer or PAGE, stored width,
#  expected fault: None, "gp" (aligned-operand #GP) or "write"[, displacement of the store from RCX])
CASES = [
    ("nt_movntps", "movntps [rcx], xmm0", LOAD_XMM0, bytes.fromhex("0f2b01"), 16, 16, None),
    ("nt_movntpd", "movntpd [rcx], xmm0", LOAD_XMM0, bytes.fromhex("660f2b01"), 32, 16, None),
    ("nt_movntdq", "movntdq [rcx], xmm0", LOAD_XMM0, bytes.fromhex("660fe701"), 0, 16, None),
    ("nt_movntps_disp", "movntps [rcx+16], xmm2", LOAD_XMM2, bytes.fromhex("0f2b5110"), 0, 16, None, 16),
    ("nt_movnti32", "movnti [rcx], eax", VALUE_RAX, bytes.fromhex("0fc301"), 8, 4, None),
    ("nt_movnti64", "movnti [rcx], rax", VALUE_RAX, bytes.fromhex("480fc301"), 24, 8, None),
    ("nt_movnti32_odd", "movnti [rcx], eax at an odd address", VALUE_RAX, bytes.fromhex("0fc301"), 3, 4, None),
    ("nt_movnti64_odd", "movnti [rcx], rax at an odd address", VALUE_RAX, bytes.fromhex("480fc301"), 41, 8, None),
    ("nt_movntps_misaligned", "movntps [rcx], xmm0 at +8", LOAD_XMM0, bytes.fromhex("0f2b01"), 8, 16, "gp"),
    ("nt_movntpd_misaligned", "movntpd [rcx], xmm0 at +4", LOAD_XMM0, bytes.fromhex("660f2b01"), 4, 16, "gp"),
    ("nt_movntdq_misaligned", "movntdq [rcx], xmm0 at +1", LOAD_XMM0, bytes.fromhex("660fe701"), 1, 16, "gp"),
    ("nt_movntps_noaccess", "movntps to PAGE_NOACCESS", LOAD_XMM0, bytes.fromhex("0f2b01"), PAGE, 16, "write"),
    ("nt_movntdq_noaccess", "movntdq to PAGE_NOACCESS", LOAD_XMM0, bytes.fromhex("660fe701"), PAGE, 16, "write"),
    ("nt_movnti64_noaccess", "movnti [rcx], rax to PAGE_NOACCESS", VALUE_RAX, bytes.fromhex("480fc301"), PAGE, 8,
     "write"),
]

TEST_MAIN = r"""
using Fn = word (*)(word, word, word);

enum Expect { NONE, GP, WRITE };

struct Case {
    const char* name;
    const char* description;
    const unsigned char* native_bytes;
    Fn lifted;
    int offset;   // of RCX in the buffer; -1: the PAGE_NOACCESS page
    int disp;     // of the store from RCX
    int width;
    Expect expect;
};

struct Fault {
    unsigned long code = 0;
    unsigned long long info0 = 0, info1 = 0;
    bool raised = false;
};

static int capture(EXCEPTION_POINTERS* pointers, Fault* fault)
{
    const EXCEPTION_RECORD* record = pointers->ExceptionRecord;
    fault->raised = true;
    fault->code = record->ExceptionCode;
    fault->info0 = record->NumberParameters > 0 ? record->ExceptionInformation[0] : ~0ull;
    fault->info1 = record->NumberParameters > 1 ? record->ExceptionInformation[1] : ~0ull;
    return EXCEPTION_EXECUTE_HANDLER;
}

static Fault call(Fn fn, word target, word source, word value)
{
    Fault fault;
    __try {
        (void)fn(target, source, value);
    } __except (capture(GetExceptionInformation(), &fault)) {
    }
    return fault;
}

static const Case cases[] = {
/*CASES*/
};

static const unsigned char SENTINEL = 0xA5;
static const unsigned long long VALUE = 0x0123456789ABCDEFull;

int main()
{
    // Page 0: the two 64-byte buffers (native at 0, lifted at 128) and the 16 source bytes at 256.
    // Page 1: PAGE_NOACCESS.
    unsigned char* base =
        static_cast<unsigned char*>(VirtualAlloc(nullptr, 8192, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    DWORD old = 0;
    if (!base || !VirtualProtect(base + 4096, 4096, PAGE_NOACCESS, &old)) {
        std::printf("skip: VirtualAlloc/VirtualProtect failed\n");
        return 77;
    }
    unsigned char* native_buf = base;
    unsigned char* lifted_buf = base + 128;
    unsigned char* source = base + 256;
    unsigned char* page = base + 4096;
    for (int k = 0; k < 16; ++k) source[k] = static_cast<unsigned char>(0x10 + 7 * k);

    int failures = 0;
    for (const Case& test : cases) {
        memset(native_buf, SENTINEL, 64);
        memset(lifted_buf, SENTINEL, 64);
        const Fn native = reinterpret_cast<Fn>(const_cast<unsigned char*>(test.native_bytes));
        const word native_target = reinterpret_cast<word>(test.offset < 0 ? page : native_buf + test.offset);
        const word lifted_target = reinterpret_cast<word>(test.offset < 0 ? page : lifted_buf + test.offset);
        const Fault nf = call(native, native_target, reinterpret_cast<word>(source), VALUE);
        const Fault lf = call(test.lifted, lifted_target, reinterpret_cast<word>(source), VALUE);

        // The reference: the stored bytes at the offset when the case does not fault, the sentinel elsewhere.
        unsigned char reference[64];
        memset(reference, SENTINEL, 64);
        if (test.expect == NONE)
            memcpy(reference + test.offset + test.disp, test.width == 16 ? source : reinterpret_cast<const unsigned char*>(&VALUE),
                   test.width);
        bool expected_fault = false;
        if (test.expect == NONE) expected_fault = !nf.raised;
        if (test.expect == GP)
            expected_fault = nf.raised && nf.code == 0xC0000005ul && nf.info0 == 0 && nf.info1 == ~0ull;
        if (test.expect == WRITE)
            expected_fault = nf.raised && nf.code == 0xC0000005ul && nf.info0 == 1 &&
                             nf.info1 >= reinterpret_cast<word>(page) && nf.info1 < reinterpret_cast<word>(page) + 16;
        const bool same_fault = nf.raised == lf.raised && nf.code == lf.code && nf.info0 == lf.info0 &&
                                (test.expect == WRITE ? lf.info1 == nf.info1 : lf.info1 == nf.info1);
        const bool native_ref = memcmp(native_buf, reference, 64) == 0;
        const bool lifted_ref = memcmp(lifted_buf, reference, 64) == 0;
        const bool ok = expected_fault && same_fault && native_ref && lifted_ref;
        std::printf("%s (%s): native fault=%d code=0x%08lx info=%llu,0x%llx lifted fault=%d code=0x%08lx "
                    "info=%llu,0x%llx buffers native=%s lifted=%s %s\n",
                    test.name, test.description, nf.raised ? 1 : 0, nf.code, nf.info0, nf.info1, lf.raised ? 1 : 0,
                    lf.code, lf.info0, lf.info1, native_ref ? "ref" : "DIFF", lifted_ref ? "ref" : "DIFF",
                    ok ? "ok" : "MISMATCH");
        if (!ok) ++failures;
    }

    // Control `noalign`: without its alignment test the lifted MOVNTPS stores at +8, where the native one faults.
    {
        memset(lifted_buf, SENTINEL, 64);
        const Fault f = call(lifted_control_noalign, reinterpret_cast<word>(lifted_buf + 8),
                             reinterpret_cast<word>(source), VALUE);
        const bool broke = !f.raised && memcmp(lifted_buf + 8, source, 16) == 0;
        std::printf("control noalign: fault=%d %s\n", f.raised ? 1 : 0,
                    broke ? "ok (differs from native)" : "MISMATCH (the misaligned cases do not test the check)");
        if (!broke) ++failures;
    }
    // Control `nostore`: with the store replaced by a NOP the reference must not match.
    {
        memset(native_buf, SENTINEL, 64);
        const Fault f = call(reinterpret_cast<Fn>(const_cast<unsigned char*>(nat_control_nostore)),
                             reinterpret_cast<word>(native_buf + 16), reinterpret_cast<word>(source), VALUE);
        const bool broke = !f.raised && memcmp(native_buf + 16, source, 16) != 0;
        std::printf("control nostore: %s\n", broke ? "ok (reference unmatched)" : "MISMATCH (reference cannot see a store)");
        if (!broke) ++failures;
    }
    // Control: a plain MOVUPS to the same misaligned target does not fault.
    {
        memset(native_buf, SENTINEL, 64);
        const Fault f = call(reinterpret_cast<Fn>(const_cast<unsigned char*>(nat_control_movups)),
                             reinterpret_cast<word>(native_buf + 8), reinterpret_cast<word>(source), VALUE);
        const bool ok = !f.raised && memcmp(native_buf + 8, source, 16) == 0;
        std::printf("control movups at +8: fault=%d %s\n", f.raised ? 1 : 0, ok ? "ok" : "MISMATCH");
        if (!ok) ++failures;
    }

    if (failures) {
        std::printf("FAIL: %d case(s)\n", failures);
        return 1;
    }
    std::printf("PASS: every non-temporal store matched native, including the alignment and access faults\n");
    return 0;
}
"""


def lifted(name: str, code: bytes, mutate=None) -> str:
    text = lower(list(Decoder(64, code)), ["rcx = a; rdx = b; r8 = c;"], {}, {}, {})
    if mutate:
        text = mutate(text)
    return f'extern "C" __declspec(dllexport) word lifted_{name}(word a, word b, word c) {{\n{text}\n}}'


def native(name: str, code: bytes) -> str:
    emitted = ", ".join("0x%02x" % byte for byte in code)
    return f'#pragma section(".text")\n__declspec(allocate(".text")) const unsigned char nat_{name}[] = {{{emitted}}};'


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "emit":
        print(__doc__)
        return 2
    output = Path(sys.argv[2])

    parts = [INTEGER_HEADER, "#include <windows.h>\n#include <cstdio>\n#include <cstring>\n"]
    rows = []
    for name, description, prologue, store, offset, width, fault, *disp in CASES:
        code = prologue + store + EPILOGUE
        parts.append(native(name, code))
        parts.append(lifted(name, code))
        expect = {None: "NONE", "gp": "GP", "write": "WRITE"}[fault]
        rows.append(f'    {{"{name}", "{description}", nat_{name}, lifted_{name}, '
                    f'{-1 if offset == PAGE else offset}, {disp[0] if disp else 0}, {width}, {expect}}},')

    movntps = LOAD_XMM0 + bytes.fromhex("0f2b01") + EPILOGUE

    def drop_align(text: str) -> str:
        # The streaming intrinsic compiles to MOVNTDQ, which faults on its own, so the control also turns it
        # into a plain store: a lowering that ignores the alignment rule.
        stream = "_mm_stream_si128(reinterpret_cast<__m128i*>(target), value);"
        if text.count(ALIGN) != 1 or text.count(stream) != 1:
            raise SystemExit("noalign control: the lifted MOVNTPS has no single alignment test and store to replace")
        return text.replace(ALIGN, "").replace(stream, "memcpy(reinterpret_cast<void*>(target), &value, 16);")

    parts.append(lifted("control_noalign", movntps, drop_align))
    parts.append(native("control_nostore", LOAD_XMM0 + bytes.fromhex("0f1f00") + EPILOGUE))   # nop dword [rax]
    parts.append(native("control_movups", LOAD_XMM0 + bytes.fromhex("0f1101") + EPILOGUE))    # movups [rcx], xmm0
    parts.append(TEST_MAIN.replace("/*CASES*/", "\n".join(rows)))
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"emitted {len(CASES)} non-temporal store cases into {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
