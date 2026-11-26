// System commit and job memory for the startup run's diagnostics: the x2w-budget heartbeat line and the first
// commit-limit failures of the physical-page pool (xmem_mapped_pages.cpp). Header only, no allocation.
#pragma once

#include <windows.h>

#include <cstdio>

namespace x2w_local {

struct SystemCommit {
    bool known = false;
    unsigned long long used_mib = 0;   // charged against the system commit limit (all processes)
    unsigned long long limit_mib = 0;  // RAM plus page files
    unsigned long long avail_mib = 0;  // limit minus used: what a new commit can still get
};

inline SystemCommit system_commit() noexcept
{
    SystemCommit commit;
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status) || status.ullAvailPageFile > status.ullTotalPageFile) return commit;
    commit.known = true;
    commit.limit_mib = status.ullTotalPageFile >> 20;
    commit.avail_mib = status.ullAvailPageFile >> 20;
    commit.used_mib = (status.ullTotalPageFile - status.ullAvailPageFile) >> 20;
    return commit;
}

// The memory limit of the job the process runs in (debug_launch.py puts the title in a kill-on-close job whose
// JobMemoryLimit is run.ps1 -MemoryGiB) and the peak commit the job has used.
struct JobMemory {
    bool known = false;    // the job could be queried (the process is in a job)
    bool limited = false;  // the job has a JobMemoryLimit
    unsigned long long limit_mib = 0;
    unsigned long long peak_mib = 0;
};

inline JobMemory job_memory() noexcept
{
    JobMemory job;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    if (!QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &info, sizeof(info), nullptr)) return job;
    job.known = true;
    job.limited = (info.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_JOB_MEMORY) != 0;
    job.limit_mib = static_cast<unsigned long long>(info.JobMemoryLimit) >> 20;
    job.peak_mib = static_cast<unsigned long long>(info.PeakJobMemoryUsed) >> 20;
    return job;
}

// True when a commit that failed was probably refused by the job limit: the job has one and its peak is within
// `slack_mib` of it (a refused request never reaches the limit itself; 256 MiB covers the largest page batch).
inline bool job_limit_reached(const JobMemory& job, unsigned long long slack_mib = 256) noexcept
{
    return job.known && job.limited && job.limit_mib > 0 && job.peak_mib + slack_mib >= job.limit_mib;
}

// "commit=<used>/<limit> avail=<free>" in MiB, or "commit=n/a avail=n/a". Returns snprintf's result.
inline int format_commit(char* out, std::size_t size, const SystemCommit& commit) noexcept
{
    if (!commit.known) return std::snprintf(out, size, "commit=n/a avail=n/a");
    return std::snprintf(out, size, "commit=%llu/%llu avail=%llu", commit.used_mib, commit.limit_mib, commit.avail_mib);
}

// "job_peak=<MiB> job_limit=<MiB>", "job_limit=none" for a job without a limit, "n/a" outside a job.
inline int format_job(char* out, std::size_t size, const JobMemory& job) noexcept
{
    if (!job.known) return std::snprintf(out, size, "job_peak=n/a job_limit=n/a");
    if (!job.limited) return std::snprintf(out, size, "job_peak=%llu job_limit=none", job.peak_mib);
    return std::snprintf(out, size, "job_peak=%llu job_limit=%llu", job.peak_mib, job.limit_mib);
}

// ERROR_COMMITMENT_LIMIT (1455: a commit limit) or ERROR_NOT_ENOUGH_MEMORY (8) from a map or commit.
inline bool is_commit_failure(unsigned long error) noexcept { return error == 1455 || error == 8; }

// The line logged at the first commit failures of the physical-page pool (xmem_mapped_pages.cpp). It names the
// job memory limit when the job is the likely cause, otherwise the system commit.
inline int format_exhausted(char* out, std::size_t size, const char* what, unsigned long error, const SystemCommit& commit,
                            const JobMemory& job, unsigned long long sections, unsigned long long slabs,
                            unsigned long long slots_in_use, unsigned long long slots_free) noexcept
{
    char state[80], jobs[64], cause[96];
    format_commit(state, sizeof(state), commit);
    format_job(jobs, sizeof(jobs), job);
    if (job_limit_reached(job))
        std::snprintf(cause, sizeof(cause), "job memory limit %llu GiB reached (run.ps1 -MemoryGiB)", (job.limit_mib + 512) >> 10);
    else
        std::snprintf(cause, sizeof(cause), "job memory limit not reached");
    return std::snprintf(out, size,
                         "x2w-xmem commit exhausted at %s (error %lu): %s; %s; system %s; pool sections=%llu slabs=%llu slots=%llu/%llu; "
                         "X2W_XMEM_LAZY_COMMIT=1 (run.ps1 -LazyMemory) keeps untouched pages out of the commit",
                         what, error, cause, jobs, state, sections, slabs, slots_in_use, slots_free);
}

}  // namespace x2w_local
