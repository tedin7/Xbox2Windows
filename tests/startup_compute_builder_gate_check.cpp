// Checks src/startup_stubs/compute_builder_gate.h without a device or a title module: when the
// opt-in console BVH builder gate allows a Dispatch, when it refuses a partial builder, and when a
// failed on-the-spot creation latches it off. Expectations are written out below, not computed with
// the header.
#include "startup_stubs/compute_builder_gate.h"

#include <cstdio>
#include <cstring>

namespace {
using x2w_startup::BuilderGateDecision;
using x2w_startup::BuilderGateState;

int failures, cases;

void expect(bool condition, const char* label)
{
    ++cases;
    if (condition) return;
    ++failures;
    std::printf("FAIL %s\n", label);
}

void expect_decision(BuilderGateState& gate, bool builder_dispatch, BuilderGateDecision expected, const char* label)
{
    std::printf("case %s\n", label);
    const BuilderGateDecision got = gate.decide(builder_dispatch);
    ++cases;
    if (got == expected) return;
    ++failures;
    std::printf("FAIL %s: expected '%s', got '%s'\n", label, x2w_startup::builder_gate_reason(expected),
                x2w_startup::builder_gate_reason(got));
}
}  // namespace

int main()
{
    using x2w_startup::is_compute_builder_module;

    expect(is_compute_builder_module(L"XboxBvhBuilder.dll"), "bare module name");
    expect(is_compute_builder_module(L"xboxbvHBuilder.DLL"), "case-insensitive module name");
    expect(is_compute_builder_module(L"C:\\Games\\XboxBvhBuilder.dll"), "backslash path");
    expect(is_compute_builder_module(L"C:/Games/XboxBvhBuilder.dll"), "forward-slash path");
    expect(!is_compute_builder_module(L"C:\\Games\\XboxBvhBuilderExtra.dll"), "extra suffix is not the builder");
    expect(!is_compute_builder_module(L"C:\\Games\\NotXboxBvhBuilder.dll"), "prefix is not the builder");
    expect(!is_compute_builder_module(L"C:\\Games\\XboxBvhBuilder.dll.bak"), "backup suffix is not the builder");
    expect(!is_compute_builder_module(L""), "empty path");
    expect(!is_compute_builder_module(L"C:\\Games\\"), "directory path");
    expect(!is_compute_builder_module(L"C:\\Games\\XboxBvhBuilder.dll\\"), "trailing slash has no basename");

    char slots[BuilderGateState::max_pipelines + 1]{};

    // Off by default: tracking is a no-op and the existing path is unchanged even with a missing set.
    BuilderGateState disabled(false);
    expect(disabled.note_pipeline(&slots[0], true, false), "disabled gate accepts a note");
    expect(disabled.tracked() == 0, "disabled gate tracks nothing");
    expect_decision(disabled, true, BuilderGateDecision::unchanged, "gate off leaves a builder Dispatch unchanged");
    expect_decision(disabled, false, BuilderGateDecision::unchanged, "gate off leaves a non-builder Dispatch unchanged");

    // Every tracked builder pipeline present: the all-or-none set is allowed.
    BuilderGateState gate(true);
    expect(gate.note_pipeline(&slots[0], true, true), "track a real builder pipeline");
    expect(gate.note_pipeline(&slots[1], true, true), "track a second real builder pipeline");
    expect(gate.tracked() == 2 && gate.with_pso() == 2, "two builder pipelines with PSOs");
    expect_decision(gate, true, BuilderGateDecision::allow, "all builder pipelines have a PSO");
    expect_decision(gate, false, BuilderGateDecision::unchanged, "a non-builder Dispatch is unchanged");

    // A non-builder pipeline never joins the tracked set and never blocks the builder.
    expect(gate.note_pipeline(&slots[2], false, false), "ignore a non-builder pipeline");
    expect(gate.tracked() == 2 && gate.with_pso() == 2, "non-builder pipeline is not tracked");
    expect_decision(gate, true, BuilderGateDecision::allow, "non-builder pipeline leaves the builder allowed");

    // One tracked builder pipeline without a PSO: refuse until the on-the-spot attempt fills it.
    expect(gate.note_pipeline(&slots[3], true, false), "track a placeholder builder pipeline");
    expect(gate.tracked() == 3 && gate.with_pso() == 2, "one builder pipeline is missing its PSO");
    expect_decision(gate, true, BuilderGateDecision::refuse_missing, "one missing builder pipeline is refused");
    gate.note_pso(&slots[3]);
    expect(gate.with_pso() == 3, "the created PSO joins the tracked set");
    expect_decision(gate, true, BuilderGateDecision::allow, "the completed builder set is allowed");

    // A repeated creation updates the entry instead of duplicating it.
    expect(gate.note_pipeline(&slots[3], true, true), "repeat an existing builder pipeline");
    expect(gate.tracked() == 3 && gate.with_pso() == 3, "the repeated pointer does not add an entry");

    // One failed on-the-spot creation latches the builder off for the rest of the run.
    BuilderGateState failed(true);
    expect(failed.note_pipeline(&slots[0], true, true), "failed set: first pipeline");
    expect(failed.note_pipeline(&slots[1], true, false), "failed set: second pipeline is missing");
    expect_decision(failed, true, BuilderGateDecision::refuse_missing, "failed set starts refused");
    failed.note_failed("desktop pipeline creation failed");
    expect(failed.off(), "a failed creation latches the gate off");
    expect(std::strcmp(failed.first_reason(), "desktop pipeline creation failed") == 0, "first reason is kept");
    expect_decision(failed, true, BuilderGateDecision::refuse_off, "the off builder stays refused");
    expect_decision(failed, true, BuilderGateDecision::refuse_off, "the refusal is permanent");
    expect_decision(failed, false, BuilderGateDecision::unchanged, "the off builder leaves other Dispatches alone");

    // An untrackable set is not silently allowed: filling the table latches the gate off.
    BuilderGateState full(true);
    for (unsigned i = 0; i < BuilderGateState::max_pipelines; ++i)
        expect(full.note_pipeline(&slots[i], true, true), "fill the builder pipeline table");
    expect(full.tracked() == BuilderGateState::max_pipelines && full.with_pso() == BuilderGateState::max_pipelines,
           "the full table is still all present");
    expect_decision(full, true, BuilderGateDecision::allow, "a full tracked set with every PSO is allowed");
    expect(!full.note_pipeline(&slots[BuilderGateState::max_pipelines], true, false), "an extra builder pipeline is refused");
    expect(full.off() && std::strcmp(full.first_reason(), "builder pipeline table is full") == 0,
           "an untrackable builder set latches off");
    expect_decision(full, true, BuilderGateDecision::refuse_off, "the untrackable builder set stays refused");

    std::printf("startup_compute_builder_gate: %d case(s), %d failure(s)\n", cases, failures);
    return failures ? 1 : 0;
}
