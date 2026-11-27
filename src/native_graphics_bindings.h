#pragma once

#include "gfx1013_compute_native.h"
#include "startup_stubs/translated_sampler_bindings.h"

namespace x2w {
// Registers were assigned by the pair linker in vertex, then pixel order.
// Preserve the guest source of each SRV and both sampler modes through PSO creation.
struct NativeGraphicsBindings {
    std::vector<ComputeViewBinding> views;
    unsigned srv_count = 0;
    x2w_startup::TranslatedSamplers samplers;

    explicit NativeGraphicsBindings(const DirectGraphicsPair &pair) {
        std::vector<ComputeViewBinding> writes;
        for (const auto *stage : {&pair.vertex, &pair.pixel}) {
            samplers.heap |= stage->sampler_heap;
            samplers.reserved_fallback |= stage->sampler_heap;
            for (const auto &v : stage->bindings) {
                auto &range = compute_uav(v.kind) ? writes : views;
                if (v.shader_register != range.size())
                    throw std::runtime_error("unsupported graphics register order");
                range.push_back(v);
            }
            for (const auto &s : stage->samplers) {
                samplers.statics.push_back(s.words);
                samplers.comparison.push_back(s.comparison);
                samplers.lod_limits.push_back(s.explicit_lod);
            }
        }
        srv_count = static_cast<unsigned>(views.size());
        views.insert(views.end(), writes.begin(), writes.end());
        if (views.size() > 32)
            throw std::runtime_error("graphics root capacity exceeded");
    }
};
}
