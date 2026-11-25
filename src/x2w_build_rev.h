#pragma once

// Source revision this binary was built from: `git rev-parse --short=8 HEAD`, with "-dirty" when tracked
// files differed from HEAD, or "unknown" without git. cmake/x2w_build_rev.cmake writes the definitions
// into a generated source on every build, so only that small file recompiles when the revision changes.
// At most 40 hex digits plus "-dirty" (46 characters).
extern const char x2w_build_rev[];
extern const wchar_t x2w_build_rev_w[];
