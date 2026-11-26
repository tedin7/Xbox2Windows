// Device-free checks for src/startup_stubs/per_reason_refusal_census.h: counts every refusal
// reason per key rather than overwriting earlier reasons with the last one.
// Compile with -DX2W_CENSUS_KEEP_LAST or -DX2W_CENSUS_KEEP_FIRST for the two negative controls
// (testing the real PerReasonRefusalCensus class under single-reason-per-key regressions).
#include "startup_stubs/per_reason_refusal_census.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(const char* what, bool ok)
{
    if (ok) return;
    ++failures;
    std::printf("FAIL %s\n", what);
}

std::vector<std::string> collect_report(x2w_startup::PerReasonRefusalCensus& census, bool only_if_changed)
{
    std::vector<std::string> lines;
    census.report([&](const char* line) { lines.emplace_back(line); }, only_if_changed);
    return lines;
}

bool contains_line(const std::vector<std::string>& lines, const std::string& expected)
{
    for (const auto& line : lines) {
        if (line == expected) return true;
    }
    return false;
}

void check_two_reasons_per_key()
{
    using namespace x2w_startup;
    constexpr std::uint64_t key1 = 0x1000000000000001ull;
    constexpr std::uint64_t key2 = 0x2000000000000002ull;
    constexpr const char* reason_a = "missing_root";
    constexpr const char* reason_b = "srv_unavailable";

    PerReasonRefusalCensus census("slot 67");
    check("first pair on key1 is detailed", census.note(key1, reason_a));
    check("repeat pair on key1 is not detailed", !census.note(key1, reason_a));
    check("second reason on key1 is detailed", census.note(key1, reason_b));
    check("repeat of second reason on key1 is not detailed", !census.note(key1, reason_b));
    check("third hit of second reason on key1 is not detailed", !census.note(key1, reason_b));

    // Interleave reasons on a second key so reason counts cannot alias across keys.
    check("key2 uav_missing is detailed", census.note(key2, "uav_missing"));
    check("key2 missing_root is detailed", census.note(key2, reason_a));
    check("key2 second uav_missing is not detailed", !census.note(key2, "uav_missing"));

    const std::vector<std::string> observed = collect_report(census, true);
    const std::string exp_summary =
        "x2w-d3d12 slot 67 refusals: total 8; distinct keys 2, distinct pairs 4 (overflow calls 0)";
    const std::string exp_k1_a = "x2w-d3d12 slot 67 refusal 1: key=1000000000000001 reason=missing_root calls=2";
    const std::string exp_k1_b = "x2w-d3d12 slot 67 refusal 2: key=1000000000000001 reason=srv_unavailable calls=3";
    const std::string exp_k2_uav = "x2w-d3d12 slot 67 refusal 3: key=2000000000000002 reason=uav_missing calls=2";
    const std::string exp_k2_root = "x2w-d3d12 slot 67 refusal 4: key=2000000000000002 reason=missing_root calls=1";

    if (!contains_line(observed, exp_k1_a) || !contains_line(observed, exp_k1_b)) {
        std::printf("expected both key1 reasons: %s calls=2 and %s calls=3\n", reason_a, reason_b);
        for (const auto& line : observed) {
            std::printf("  observed: %s\n", line.c_str());
        }
    }

    check("summary includes both keys and all four pairs", contains_line(observed, exp_summary));
    check("key1 keeps first reason missing_root with calls=2", contains_line(observed, exp_k1_a));
    check("key1 keeps second reason srv_unavailable with calls=3", contains_line(observed, exp_k1_b));
    check("key2 keeps interleaved uav_missing with calls=2", contains_line(observed, exp_k2_uav));
    check("key2 keeps interleaved missing_root with calls=1", contains_line(observed, exp_k2_root));
}

void check_limits_and_periodic_deltas()
{
    using namespace x2w_startup;
    PerReasonRefusalCensus census;
    check("empty periodic report is skipped", collect_report(census, true).empty());
    check("empty forced report still emits summary", collect_report(census, false).size() == 1);

    // Null/empty reasons normalize to unknown and newlines become spaces.
    check("null reason is detailed", census.note(1, nullptr));
    check("empty reason merges into unknown", !census.note(1, ""));
    check("newline sanitized reason is detailed", census.note(2, "line1\nline2"));

    for (unsigned i = 2; i < PerReasonRefusalCensus::detail_entries; ++i) {
        check("entries up to detail_entries are detailed", census.note(100 + i, "r"));
    }
    check("ninth distinct entry is not detailed", !census.note(999, "r"));

    check("capacity is 1024 pairs", PerReasonRefusalCensus::max_entries == 1024);
    while (census.entry_count() < PerReasonRefusalCensus::max_entries) {
        census.note(1000 + census.entry_count(), "fill");
    }
    check("entry past max_entries is not detailed", !census.note(5000, "overflow"));
    check("second call past max_entries counts as overflow", !census.note(5001, "overflow"));
    check("overflow count is two", census.overflow_calls() == 2);

    const auto late_key = 1000 + census.entry_count() - 1;
    for (int i = 0; i < 4; ++i) census.note(late_key, "fill");
    const auto full = collect_report(census, true);
    check("report has summary, top 64 pairs and omitted count", full.size() == 66);
    if (full.size() == 66) {
        check("late hot pair is first",
              full[1] == "x2w-d3d12 per-reason refusal 1024: key=00000000000007e7 reason=fill calls=5");
        check("ties retain insertion order", full[64].find("refusal 63:") != std::string::npos);
        check("omitted count excludes overflow", full[65] == "x2w-d3d12 per-reason refusals omitted: pairs=960");
    }
    check("unknown normalized and merged",
          contains_line(full, "x2w-d3d12 per-reason refusal 1: key=0000000000000001 reason=unknown calls=2"));
    check("newline replaced by space",
          contains_line(full, "x2w-d3d12 per-reason refusal 2: key=0000000000000002 reason=line1 line2 calls=1"));

    check("unchanged periodic report is skipped", collect_report(census, true).empty());
    census.note(1, nullptr);
    const auto delta = collect_report(census, true);
    check("changed periodic report keeps the complete ranking",
          delta.size() == 66 &&
              delta[2] == "x2w-d3d12 per-reason refusal 1: key=0000000000000001 reason=unknown calls=3");

    census.note(2, "line1\nline2");
    const auto forced = collect_report(census, false);
    check("forced report uses the same top 64", forced.size() == 66);
    check("forced report does not suppress the next periodic report", collect_report(census, true) == forced);
}

}  // namespace

int main(int argc, char**)
{
    if (argc != 1) {
        std::fprintf(stderr, "usage: startup_per_reason_refusal_census_check\n");
        return 2;
    }

    check_two_reasons_per_key();
    check_limits_and_periodic_deltas();

    if (failures) {
        std::printf("startup_per_reason_refusal_census: FAIL (%d checks)\n", failures);
        return 1;
    }
    std::printf("startup_per_reason_refusal_census: PASS (distinct keys=2, pairs=4, detail=8, max=%u)\n",
                x2w_startup::PerReasonRefusalCensus::max_entries);
    return 0;
}
