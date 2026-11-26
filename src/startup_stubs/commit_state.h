// System commit charge for the startup run's diagnostics: the x2w-budget heartbeat line and the first
// commit-limit failure of the physical-page pool (xmem_mapped_pages.cpp). Header only, no allocation.
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

// "commit=<used>/<limit> avail=<free>" in MiB, or "commit=n/a avail=n/a". Returns snprintf's result.
inline int format_commit(char* out, std::size_t size, const SystemCommit& commit) noexcept
{
    if (!commit.known) return std::snprintf(out, size, "commit=n/a avail=n/a");
    return std::snprintf(out, size, "commit=%llu/%llu avail=%llu", commit.used_mib, commit.limit_mib, commit.avail_mib);
}

// ERROR_COMMITMENT_LIMIT (1455: the system commit limit) or ERROR_NOT_ENOUGH_MEMORY (8) from a map or commit.
inline bool is_commit_failure(unsigned long error) noexcept { return error == 1455 || error == 8; }

// The line logged at the first commit failures of the physical-page pool (xmem_mapped_pages.cpp).
inline int format_exhausted(char* out, std::size_t size, const char* what, unsigned long error, const SystemCommit& commit,
                            unsigned long long sections, unsigned long long slabs, unsigned long long slots_in_use,
                            unsigned long long slots_free) noexcept
{
    char state[80];
    format_commit(state, sizeof(state), commit);
    return std::snprintf(out, size,
                         "x2w-xmem commit exhausted at %s (error %lu): system %s; pool sections=%llu slabs=%llu slots=%llu/%llu; "
                         "X2W_XMEM_LAZY_COMMIT=1 (run.ps1 -LazyMemory) keeps untouched pages out of the commit",
                         what, error, state, sections, slabs, slots_in_use, slots_free);
}

}  // namespace x2w_local
