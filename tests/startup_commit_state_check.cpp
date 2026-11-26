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

    JobMemory capped;  // a 14 GiB job whose peak is at its limit
    capped.known = capped.limited = true;
    capped.limit_mib = 14336;
    capped.peak_mib = 14250;
    JobMemory roomy = capped;  // the same job with a gigabyte to spare
    roomy.peak_mib = 13000;
    JobMemory unlimited;
    unlimited.known = true;
    unlimited.peak_mib = 900;
    format_job(text, sizeof(text), capped);
    expect_text(text, "job_peak=14250 job_limit=14336", "limited job text");
    format_job(text, sizeof(text), unlimited);
    expect_text(text, "job_peak=900 job_limit=none", "unlimited job text");
    format_job(text, sizeof(text), JobMemory{});
    expect_text(text, "job_peak=n/a job_limit=n/a", "no job text");
    expect(job_limit_reached(capped) && !job_limit_reached(roomy) && !job_limit_reached(unlimited) &&
               !job_limit_reached(JobMemory{}),
           "the job limit is the cause only for a limited job near its limit");

    format_exhausted(text, sizeof(text), "XMemMapPhysicalPagesScatter", 1455, known, capped, 113609, 3, 113000, 609);
    expect_text(text,
                "x2w-xmem commit exhausted at XMemMapPhysicalPagesScatter (error 1455): job memory limit 14 GiB reached "
                "(run.ps1 -MemoryGiB); job_peak=14250 job_limit=14336; system commit=14335/16100 avail=1765; "
                "pool sections=113609 slabs=3 slots=113000/609; "
                "X2W_XMEM_LAZY_COMMIT=1 (run.ps1 -LazyMemory) keeps untouched pages out of the commit",
                "exhausted line, job limit reached");
    format_exhausted(text, sizeof(text), "XMemMapPhysicalPagesScatter", 1455, known, roomy, 113609, 3, 113000, 609);
    expect(std::strstr(text, ": job memory limit not reached; job_peak=13000 job_limit=14336; system commit=14335/16100") != nullptr,
           "exhausted line, system commit is the cause");

    // A real job: the process joins one with a 4 GiB memory limit and job_memory() reads it back. (Skipped when the
    // machine does not let this process join another job.)
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY;
    info.JobMemoryLimit = static_cast<SIZE_T>(4096) << 20;
    if (job && SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info, sizeof(info)) &&
        AssignProcessToJobObject(job, GetCurrentProcess())) {
        const JobMemory read = job_memory();
        // Wine accepts the limit but does not store it: only Windows must read it back.
        const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
        if (wine) {
            std::printf("note: Wine does not keep JobMemoryLimit, real-job case skipped\n");
        } else {
            expect(read.known && read.limited && read.limit_mib == 4096 && read.peak_mib < 4096, "a real job is read back");
            expect(!job_limit_reached(read), "a fresh process is far from a 4 GiB job limit");
        }
    } else {
        std::printf("note: could not join a test job (error %lu), real-job case skipped\n", GetLastError());
    }

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
