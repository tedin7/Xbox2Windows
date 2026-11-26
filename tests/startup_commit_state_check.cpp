// Check src/startup_stubs/commit_state.h: the text of the x2w-budget commit fields and of the commit-exhausted line,
// which commit errors count, and that the system commit this machine reports is consistent.
#include "startup_stubs/commit_state.h"

#include <cstdio>
#include <cstring>

namespace {
int failures = 0;

void expect(bool ok, const char* what)
{
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

void expect_text(const char* got, const char* want, const char* what)
{
    if (std::strcmp(got, want) != 0) {
        std::printf("FAIL %s: got [%s] want [%s]\n", what, got, want);
        ++failures;
    }
}
}  // namespace

int main()
{
    using namespace x2w_local;
    char text[400];

    SystemCommit known;
    known.known = true;
    known.used_mib = 14335;
    known.limit_mib = 16100;
    known.avail_mib = 1765;
    format_commit(text, sizeof(text), known);
    expect_text(text, "commit=14335/16100 avail=1765", "known commit text");
    format_commit(text, sizeof(text), SystemCommit{});
    expect_text(text, "commit=n/a avail=n/a", "unknown commit text");

    format_exhausted(text, sizeof(text), "XMemMapPhysicalPagesScatter", 1455, known, 113609, 3, 113000, 609);
    expect_text(text,
                "x2w-xmem commit exhausted at XMemMapPhysicalPagesScatter (error 1455): system commit=14335/16100 avail=1765; "
                "pool sections=113609 slabs=3 slots=113000/609; "
                "X2W_XMEM_LAZY_COMMIT=1 (run.ps1 -LazyMemory) keeps untouched pages out of the commit",
                "exhausted line");

    expect(is_commit_failure(1455) && is_commit_failure(8), "commit limit and not enough memory count");
    expect(!is_commit_failure(0) && !is_commit_failure(87) && !is_commit_failure(5) && !is_commit_failure(487),
           "other errors do not");

    const SystemCommit now = system_commit();
    expect(now.known, "this machine reports its commit");
    expect(now.limit_mib > 0 && now.used_mib > 0 && now.used_mib <= now.limit_mib && now.avail_mib <= now.limit_mib,
           "commit numbers are within the limit");
    expect(now.used_mib + now.avail_mib + 1 >= now.limit_mib && now.used_mib + now.avail_mib <= now.limit_mib + 1,
           "used plus available is the limit (to a MiB)");

    std::printf("startup_commit_state: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
