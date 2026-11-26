#!/usr/bin/env python3
"""Emit the reserved/hint-NOP differential fixture (issue #1476).

For the same inputs, compare what the host executes for a hint-NOP encoding against what the shared
lowering produces for it:

  native  - the exact bytes in an executable section, called directly.
  lifted  - the same instructions through tools/ffx_version_query.lower(), compiled alongside.

The prologue sets RAX = a and the arithmetic flags from `add rax, b`; every case then executes the encoding
under test and stores the flags in AH with LAHF, so a changed register or flag shows up as a different
return value. The memory-form cases are handed a pointer to a `PAGE_NOACCESS` page as the base register:
if anything evaluated the operand the call would fault instead of returning, so returning is part of the
assertion. Two controls make that assertion meaningful: a real load from the same page must fault (so the
harness can see a memory access at all), and a real register write must change the result (so equal results
are informative).

usage: hint_nop_fixture.py emit <output.cpp>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from iced_x86 import Decoder
from ffx_version_query import INTEGER_HEADER, lower

# Prologue: mov rax, rcx ; add rax, rdx  (sets flags from a + b), epilogue: lahf ; ret.
PROLOGUE = bytes.fromhex("4889c84801d0")
EPILOGUE = bytes.fromhex("9fc3")

# (export name, description, bytes of the instruction under test, hands it the inaccessible pointer)
CASES = [
    ("hn_0f19_reg", "0F 19 register form", bytes.fromhex("0f19c0"), False),
    ("hn_0f19_mem", "0F 19 [rax]", bytes.fromhex("0f1900"), True),
    ("hn_0f1d_mem", "0F 1D [rax]", bytes.fromhex("0f1d00"), True),
    ("hn_0f1e_mem", "0F 1E [rax]", bytes.fromhex("0f1e00"), True),
    ("hn_0f1f_disp8", "0F 1F [rax+8]", bytes.fromhex("0f1f4008"), True),
    # The prefixed and non-32-bit operand-size forms are accepted too; the Python check covers them, and
    # these run them natively against the lifted body on Windows.
    ("hn_f3_0f1e", "F3 0F 1E [rax]", bytes.fromhex("f30f1e00"), True),
    ("hn_0f19_16", "66 0F 19 [rax], 16-bit operand size", bytes.fromhex("660f1900"), True),
    ("hn_0f1e_16", "66 0F 1E [rax], 16-bit operand size", bytes.fromhex("660f1e00"), True),
    ("hn_0f19_64", "48 0F 19 [rax], 64-bit operand size", bytes.fromhex("480f1900"), True),
    ("hn_0f1e_64", "48 0F 1E [rax], 64-bit operand size", bytes.fromhex("480f1e00"), True),
    ("hn_endbr64", "F3 0F 1E FA (endbr64)", bytes.fromhex("f30f1efa"), False),
    ("hn_endbr32", "F3 0F 1E FB (endbr32)", bytes.fromhex("f30f1efb"), False),
]

TEST_MAIN = r"""
using Fn = word (*)(word, word, word);

struct Case {
    const char* name;
    const char* description;
    const unsigned char* native_bytes;
    Fn lifted;
    bool inaccessible;
};

static const Case cases[] = {
/*CASES*/
};

int main()
{
    // One PAGE_NOACCESS page: any operand evaluation faults instead of returning.
    unsigned char* page =
        static_cast<unsigned char*>(VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_NOACCESS));
    if (!page) {
        std::printf("skip: VirtualAlloc failed\n");
        return 77;
    }
    int failures = 0;
    for (const Case& test : cases) {
        const Fn native = reinterpret_cast<Fn>(const_cast<unsigned char*>(test.native_bytes));
        const word a = test.inaccessible ? reinterpret_cast<word>(page) : 0x0123456789ABCDEFull;
        // The memory forms address [base + b], i.e. page+17 or page+25 here, which is still inside the one
        // inaccessible page, so an evaluated operand would fault rather than read something mapped.
        const word b = 17;
        word native_out = 0;
        bool faulted = false;
        __try {
            native_out = native(a, b, 0);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            faulted = true;
        }
        // The lifted body is called under the same guard: a lowering that evaluated the operand must be
        // reported as a mismatch, not crash the fixture.
        word lifted_out = 0;
        bool lifted_faulted = false;
        __try {
            lifted_out = test.lifted(a, b, 0);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            lifted_faulted = true;
        }
        const bool ok = !faulted && !lifted_faulted && native_out == lifted_out;
        std::printf("%s (%s): native=0x%llx lifted=0x%llx faulted=%d lifted_faulted=%d %s\n", test.name,
                    test.description, static_cast<unsigned long long>(native_out),
                    static_cast<unsigned long long>(lifted_out), faulted ? 1 : 0, lifted_faulted ? 1 : 0,
                    ok ? "ok" : "MISMATCH");
        if (!ok) ++failures;
    }

    // Control 1: reading the same inaccessible page must fault, otherwise "did not fault" above proves
    // nothing about whether this harness can see a memory access at all.
    bool load_faulted = false;
    __try {
        volatile unsigned long value = *reinterpret_cast<volatile unsigned long*>(page);
        (void)value;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        load_faulted = true;
    }
    std::printf("control load from PAGE_NOACCESS faulted=%d %s\n", load_faulted ? 1 : 0,
                load_faulted ? "ok" : "MISMATCH (harness cannot see memory access)");
    if (!load_faulted) ++failures;

    // Control 2: a real register write, run with the same inputs as the first case, must move the result.
    // Otherwise "native equals lifted" would hold even for a fixture that executes nothing meaningful.
    word hint_out = 0, write_out = 0;
    __try {
        hint_out = reinterpret_cast<Fn>(const_cast<unsigned char*>(cases[0].native_bytes))(0x0123456789ABCDEFull, 17, 0);
        write_out = reinterpret_cast<Fn>(const_cast<unsigned char*>(nat_control_write))(0x0123456789ABCDEFull, 17, 0);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        write_out = hint_out;
    }
    const bool register_control = write_out != hint_out;
    std::printf("control register write: hint=0x%llx write=0x%llx %s\n", static_cast<unsigned long long>(hint_out),
                static_cast<unsigned long long>(write_out), register_control ? "ok" : "MISMATCH");
    if (!register_control) ++failures;

    if (failures) {
        std::printf("FAIL: %d case(s)\n", failures);
        return 1;
    }
    std::printf("PASS: every hint NOP matched its lifted body without evaluating the operand\n");
    return 0;
}
"""


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "emit":
        print(__doc__)
        return 2
    output = Path(sys.argv[2])

    parts = [INTEGER_HEADER, "#include <windows.h>\n#include <cstdio>\n"]
    rows = []
    for name, description, case_bytes, inaccessible in CASES:
        code = PROLOGUE + case_bytes + EPILOGUE
        lifted = lower(list(Decoder(64, code)), ["rcx = a; rdx = b; r8 = c;"], {}, {}, {})
        emitted = ", ".join("0x%02x" % byte for byte in code)
        parts.append(f'#pragma section(".text")\n__declspec(allocate(".text")) const unsigned char nat_{name}[] = {{{emitted}}};')
        parts.append(f'extern "C" __declspec(dllexport) word lifted_{name}(word a, word b, word c) {{\n{lifted}\n}}')
        rows.append(f'    {{"{name}", "{description}", nat_{name}, lifted_{name}, {"true" if inaccessible else "false"}}},')

    control = bytes.fromhex("b8785634129fc3")  # mov eax, 0x12345678 ; lahf ; ret
    control_emitted = ", ".join("0x%02x" % byte for byte in control)
    parts.append(f'#pragma section(".text")\n__declspec(allocate(".text")) const unsigned char nat_control_write[] = {{{control_emitted}}};')
    parts.append(TEST_MAIN.replace("/*CASES*/", "\n".join(rows)))
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"emitted {len(CASES)} hint-NOP cases into {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
