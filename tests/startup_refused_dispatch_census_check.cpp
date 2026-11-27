#include "../src/startup_stubs/refused_dispatch_census.h"

#include <cstdio>
#include <string>
#include <vector>

using x2w_startup::mesh_dispatch_groups;
using x2w_startup::RefusedDispatchCensus;

int main()
{
    RefusedDispatchCensus compute(RefusedDispatchCensus::Kind::compute);
    RefusedDispatchCensus mesh(RefusedDispatchCensus::Kind::mesh57);
    std::vector<std::string> lines;
    const auto capture = [&](const char* line) { lines.emplace_back(line); };
    const auto expect = [](bool passed, const char* label) {
        if (!passed) std::fprintf(stderr, "FAIL: %s\n", label);
        return passed;
    };
    bool ok = true;

    compute.report(capture);
    ok &= expect(lines.empty(), "empty census stays silent");
    ok &= expect(compute.note(1, "no translation"), "compute report at 1");
    ok &= expect(compute.note(1, "binding failed"), "compute report at 2");
    ok &= expect(!compute.note(2, "no translation"), "no compute report at 3");
    compute.report(capture);
    ok &= expect(lines == std::vector<std::string>{
        "x2w-render compute refused by pipeline key: 2 keys, 3 dispatches",
        "x2w-render compute refused key 0000000000000001: 2 dispatches, first refusal: no translation",
        "x2w-render compute refused key 0000000000000002: 1 dispatches, first refusal: no translation",
    }, "final compute counts and first reason");

    lines.clear();
    ok &= expect(mesh.note(1, "no matching pipeline", mesh_dispatch_groups(2, 3, 5)), "mesh report at 1");
    ok &= expect(mesh.note(1, "later reason", mesh_dispatch_groups(4, 2, 3)), "mesh report at 2");
    ok &= expect(!mesh.note(2, "zero grid", mesh_dispatch_groups(0, 4, 2)), "no mesh report at 3");
    ok &= expect(mesh.note(1, "unreadable record"), "mesh report at 4");
    ok &= expect(!mesh.note(2, "invalid grid", mesh_dispatch_groups(65535, 65535, 65535)), "no mesh report at 5");
    mesh.report(capture);
    ok &= expect(lines == std::vector<std::string>{
        "x2w-render mesh57 refused by pipeline key: 2 keys, 5 dispatches",
        "x2w-render mesh57 refused key 0000000000000001: 3 dispatches, 54 groups, 1 unknown group counts, first refusal: no matching pipeline",
        "x2w-render mesh57 refused key 0000000000000002: 2 dispatches, 0 groups, 1 unknown group counts, first refusal: zero grid",
    }, "mesh57 groups sum X*Y*Z (30 + 24), with zero and unknown tracked separately");

    for (std::uint64_t key = 100; key > 40; --key) mesh.note(key, "no matching pipeline", 1);
    lines.clear();
    mesh.report(capture);
    ok &= expect(lines.size() == 41 && lines.back().find("key 000000000000004e:") != std::string::npos,
                 "top 40 keys, deterministic tie breaking");
    return ok ? 0 : 1;
}
