// Every refusal of format_compute_runtime_manifest names its reason, and an accepted program's manifest is unchanged.
#include "direct_compute_cache.h"
#include "native_compute_output.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <string>

using namespace x2w;

static ComputeViewBinding view(ComputeViewKind kind, unsigned reg, unsigned root = 1, unsigned offset = 0) {
    ComputeViewBinding b{};
    b.kind = kind;
    b.root_parameter = root;
    b.byte_offset = offset;
    b.shader_register = reg;
    return b;
}

static NativeComputeShader base() {
    NativeComputeShader p{};
    p.group = {8, 4, 1};
    p.constants.push_back({0, 0, 2});
    p.bindings.push_back(view(ComputeViewKind::buffer_srv, 0, 1, 64));
    p.bindings.push_back(view(ComputeViewKind::raw_buffer_uav, 0, 1, 32));
    return p;
}

int main() {
    int failures = 0;
    std::string reason = "unset";
    const auto accepted = format_compute_runtime_manifest(base(), &reason);
    const std::string expected = "x2w-manifest 2\nstage cs\ngroup 8 4 1\nud 2\nudsrc 0 rootconst 0 0\n"
                                 "udsrc 1 rootconst 0 1\nsrv 0 buf table 1 64\nuav 0 table 1 32\nnative " +
                                 std::to_string(native_compute_output_revision) + "\n";
    if (!accepted || *accepted != expected || reason != "unset") {
        std::printf("FAIL accepted program: manifest %s, reason \"%s\"\n", accepted ? accepted->c_str() : "(none)",
                    reason.c_str());
        ++failures;
    }

    using K = ComputeViewKind;
    const struct {
        const char *expected;
        std::function<void(NativeComputeShader &)> change;
    } cases[] = {
        {"group size has a zero dimension", [](auto &p) { p.group[2] = 0; }},
        {"root constants above root parameter 15", [](auto &p) { p.constants[0].root_parameter = 16; }},
        {"more than 8 root constants in one root parameter", [](auto &p) { p.constants[0].count = 9; }},
        {"more than 32 user data words", [](auto &p) {
             p.constants.assign(4, {0, 0, 8});
             p.bindings[0].kind = K::raw_vector_srv;
             p.stride_srvs = {0};
         }},
        {"resource heap view", [](auto &p) { p.bindings.push_back(view(K::heap_buffer_srv, 1, compute_resource_heap)); }},
        {"sampler heap", [](auto &p) { p.bindings.push_back(view(K::heap_sampler, 1, compute_sampler_heap)); }},
        {"view above root parameter 15", [](auto &p) { p.bindings[1].root_parameter = 16; }},
        {"SRV registers not contiguous", [](auto &p) { p.bindings[0].shader_register = 1; }},
        {"UAV registers not contiguous", [](auto &p) { p.bindings[1].shader_register = 1; }},
        {"SRV kind root_buffer_srv", [](auto &p) { p.bindings[0].kind = K::root_buffer_srv; }},
        {"SRV kind texture_srv cube", [](auto &p) {
             p.bindings[0].kind = K::texture_srv;
             p.bindings[0].image_dimension = 3;
         }},
        {"SRV kind texture_srv 2d array", [](auto &p) {
             p.bindings[0].kind = K::texture_srv;
             p.bindings[0].array_count = 4;
         }},
        {"UAV kind typed_buffer_uav", [](auto &p) { p.bindings[1].kind = K::typed_buffer_uav; }},
        {"UAV kind texture_uav cube", [](auto &p) {
             p.bindings[1].kind = K::texture_uav;
             p.bindings[1].image_dimension = 3;
         }},
        {"more than 32 SRVs", [](auto &p) {
             for (unsigned r = 1; r < 33; ++r)
                 p.bindings.push_back(view(K::buffer_srv, r, 1, 64 + 32 * r));
         }},
        {"more than 8 UAVs", [](auto &p) {
             for (unsigned r = 1; r < 9; ++r)
                 p.bindings.push_back(view(K::raw_buffer_uav, r, 2, 32 * r));
         }},
        {"comparison sampler", [](auto &p) { p.samplers.push_back({{0, 0, 0, 0}, true, true}); }},
        {"stride words from a view that is not one raw vector SRV", [](auto &p) { p.stride_srvs = {0}; }},
    };
    for (const auto &c : cases) {
        auto program = base();
        c.change(program);
        reason.clear();
        const auto manifest = format_compute_runtime_manifest(program, &reason);
        const bool ok = !manifest && reason == c.expected;
        std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", c.expected, ok ? "" : " | got: ",
                    ok ? "" : manifest ? "a manifest" : reason.c_str());
        failures += !ok;
    }
    // A texture SRV is published under the name of its dimension: the line the runtime's reader takes.
    const struct {
        unsigned image_dimension;
        const char *line;
    } published[] = {
        {0, "srv 0 tex1d table 1 64\n"},
        {1, "srv 0 tex2d table 1 64\n"},
        {2, "srv 0 tex3d table 1 64\n"},
        {5, "srv 0 tex2da table 1 64\n"},
    };
    for (const auto &c : published) {
        auto program = base();
        program.bindings[0].kind = K::texture_srv;
        program.bindings[0].image_dimension = c.image_dimension;
        reason = "unset";
        const auto manifest = format_compute_runtime_manifest(program, &reason);
        std::string wanted = expected;
        wanted.replace(wanted.find("srv 0 buf table 1 64\n"), std::strlen("srv 0 buf table 1 64\n"), c.line);
        const bool ok = manifest && *manifest == wanted && reason == "unset";
        std::printf("%s publishes %s", ok ? "ok  " : "FAIL", c.line);
        if (!ok)
            std::printf(" | got: %s\n", manifest ? manifest->c_str() : reason.c_str());
        failures += !ok;
    }
    std::printf("%d of %zu manifest reasons wrong\n", failures, std::size(cases) + std::size(published) + 1);
    return failures ? 1 : 0;
}
