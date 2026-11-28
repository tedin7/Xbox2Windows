#pragma once

namespace x2w_startup {

enum class DebugLevel { off, layer, gpu_validation };

constexpr DebugLevel d3d12_debug_level(char first)
{
    switch (first) {
    case '1': return DebugLevel::layer;
    case '2': return DebugLevel::gpu_validation;
    default: return DebugLevel::off;
    }
}

}  // namespace x2w_startup
