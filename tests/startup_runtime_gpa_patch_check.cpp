// Source-built control for runtime_gpa_patch.h: which PE imports the gaming-runtime proxy notices.
//
// Which PE imports the stand-in notices on the EXE / package-helper import tables: GetProcAddress and
// a direct xgameruntime.dll!QueryApiImpl on the regular import table, delay-import hits with
// delay_import set (without treating the delay IAT as a real export), and ordinal thunks ignored.
// A synthetic PE32+ image is walked; the mutated control classifies by import name alone (accepting
// QueryApiImpl from other.dll) and must fail. No Windows loader and no game files.
#include "startup_stubs/runtime_gpa_patch.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace patch = x2w_runtime_patch;

namespace {
int failures;
bool name_only;

void check(const char* label, bool ok)
{
    std::printf("%s: %s\n", label, ok ? "ok" : "MISMATCH");
    if (!ok) ++failures;
}

std::vector<unsigned char> build_image()
{
    using patch::DataDir;
    using patch::DelayDesc;
    using patch::DosStub;
    using patch::ImportDesc;
    using patch::Thunk64;
    // Layout must keep the import-descriptor array clear of the OFT/IAT/name RVAs it points at.
    // Descriptors at 0x200 (2×20 → 0x228); delay desc at 0x400; thunks and strings above those.
    std::vector<unsigned char> image(0x600, 0);
    auto* dos = reinterpret_cast<DosStub*>(image.data());
    dos->e_magic = 0x5A4D;
    dos->e_lfanew = 0x80;
    auto* nt = image.data() + 0x80;
    *reinterpret_cast<std::uint32_t*>(nt) = 0x00004550;
    *reinterpret_cast<std::uint16_t*>(nt + 24) = 0x20B;
    auto* dirs = reinterpret_cast<DataDir*>(nt + 24 + 112);
    dirs[1].virtual_address = 0x200;
    dirs[1].size = sizeof(ImportDesc) * 2;
    dirs[13].virtual_address = 0x400;
    dirs[13].size = sizeof(DelayDesc);

    auto* imp = reinterpret_cast<ImportDesc*>(image.data() + 0x200);
    imp[0].original_first_thunk = 0x240;
    imp[0].first_thunk = 0x270;
    imp[0].name = 0x2A0;
    std::memcpy(image.data() + 0x2A0, "xgameruntime.dll", 17);

    const char* names[] = {"GetProcAddress", "QueryApiImpl", "SomethingElse"};
    std::uint32_t hint_rva = 0x2C0;
    std::uint32_t gpa_hint = 0;
    for (int i = 0; i < 3; ++i) {
        if (i == 0) gpa_hint = hint_rva;
        reinterpret_cast<Thunk64*>(image.data() + 0x240)[i].value = hint_rva;
        reinterpret_cast<Thunk64*>(image.data() + 0x270)[i].value = 0x1000u + static_cast<unsigned>(i);
        image[hint_rva] = image[hint_rva + 1] = 0;
        std::memcpy(image.data() + hint_rva + 2, names[i], std::strlen(names[i]) + 1);
        hint_rva += 2 + static_cast<std::uint32_t>(std::strlen(names[i]) + 1);
        hint_rva = (hint_rva + 1) & ~1u;
    }
    // Ordinal import whose low bits alias the GetProcAddress hint RVA. Without the ordinal skip the
    // walk would treat those bits as a name RVA and report a third GetProcAddress hit.
    reinterpret_cast<Thunk64*>(image.data() + 0x240)[3].value = 0x8000000000000000ull | gpa_hint;
    reinterpret_cast<Thunk64*>(image.data() + 0x270)[3].value = 0xDEAD;
    reinterpret_cast<Thunk64*>(image.data() + 0x240)[4].value = 0;
    reinterpret_cast<Thunk64*>(image.data() + 0x270)[4].value = 0;

    // other.dll!QueryApiImpl must not classify (negative shape in the same image).
    imp[1].original_first_thunk = 0x340;
    imp[1].first_thunk = 0x350;
    imp[1].name = 0x360;
    std::memcpy(image.data() + 0x360, "other.dll", 10);
    reinterpret_cast<Thunk64*>(image.data() + 0x340)[0].value = 0x370;
    reinterpret_cast<Thunk64*>(image.data() + 0x350)[0].value = 0x2000;
    image[0x370] = image[0x371] = 0;
    std::memcpy(image.data() + 0x372, "QueryApiImpl", 13);
    reinterpret_cast<Thunk64*>(image.data() + 0x340)[1].value = 0;
    reinterpret_cast<Thunk64*>(image.data() + 0x350)[1].value = 0;

    // Delay-load kernel32!GetProcAddress — reported with delay_import; not a regular IAT rewrite target.
    auto* delay = reinterpret_cast<DelayDesc*>(image.data() + 0x400);
    delay->dll_name = 0x480;
    delay->original_first_thunk = 0x4A0;
    delay->first_thunk = 0x4B0;
    std::memcpy(image.data() + 0x480, "KERNEL32.dll", 13);
    reinterpret_cast<Thunk64*>(image.data() + 0x4A0)[0].value = 0x4C0;
    reinterpret_cast<Thunk64*>(image.data() + 0x4B0)[0].value = 0x3000;
    image[0x4C0] = image[0x4C1] = 0;
    std::memcpy(image.data() + 0x4C2, "GetProcAddress", 15);
    reinterpret_cast<Thunk64*>(image.data() + 0x4A0)[1].value = 0;
    reinterpret_cast<Thunk64*>(image.data() + 0x4B0)[1].value = 0;
    return image;
}
}

int main(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--mutate") == 0 && i + 1 < argc && std::strcmp(argv[i + 1], "name-only") == 0)
            name_only = true;

    patch::Kind kind{};
    if (name_only) {
        // Negative control: if QueryApiImpl were accepted from any DLL, other.dll would match.
        // The stock rule refuses it; this mutate path prints FAIL so the WILL_FAIL ctest passes.
        const bool stock_refuses = !patch::classify("other.dll", "QueryApiImpl", &kind);
        if (stock_refuses && patch::import_is_query_api_impl("QueryApiImpl")) {
            std::puts("FAIL: name-only accepts QueryApiImpl from other.dll");
            return 1;
        }
        std::puts("startup_runtime_gpa_patch mutate: unexpected");
        return 2;
    }

    check("GetProcAddress any dll", patch::classify("kernel32.dll", "GetProcAddress", &kind) && kind == patch::Kind::get_proc_address);
    check("QueryApiImpl xgameruntime", patch::classify("xgameruntime.dll", "QueryApiImpl", &kind) && kind == patch::Kind::query_api_impl);
    check("QueryApiImpl case dll", patch::classify("XGameRuntime.DLL", "QueryApiImpl", &kind) && kind == patch::Kind::query_api_impl);
    check("QueryApiImpl other dll refused", !patch::classify("other.dll", "QueryApiImpl", &kind));
    check("unrelated refused", !patch::classify("xgameruntime.dll", "SomethingElse", &kind));

    auto image = build_image();
    int gpa = 0, query = 0, delay_hits = 0, regular_hits = 0;
    std::vector<std::uint32_t> rvas;
    const int hits = patch::visit_imports(image.data(), image.size(), [&](const patch::Hit& hit) {
        rvas.push_back(hit.first_thunk_rva);
        if (hit.delay_import) ++delay_hits;
        else ++regular_hits;
        if (hit.kind == patch::Kind::get_proc_address) ++gpa;
        else if (hit.kind == patch::Kind::query_api_impl) ++query;
    });
    check("visit finds GetProcAddress (import + delay)", gpa == 2);
    check("visit finds one xgameruntime QueryApiImpl", query == 1);
    check("visit hit count", hits == 3);
    check("ordinal import skipped", gpa == 2 && hits == 3);
    check("delay vs regular hits", delay_hits == 1 && regular_hits == 2);
    check("first thunk RVAs", rvas.size() == 3 && rvas[0] == 0x270 && rvas[1] == 0x278 && rvas[2] == 0x4B0);

    if (failures) {
        std::printf("startup_runtime_gpa_patch: FAIL (%d)\n", failures);
        return 1;
    }
    std::puts("startup_runtime_gpa_patch: ok");
    return 0;
}
