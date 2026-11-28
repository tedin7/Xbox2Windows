#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace x2w {
struct ClearBinding {
    unsigned parameter;
    unsigned byte_offset;
};
struct ComputeClear {
    std::string hlsl;
    // Entry i must bind the guest's 256-element R32_UINT UAV to register ui.
    std::vector<ClearBinding> bindings;
};
ComputeClear replay_group_clear(const std::filesystem::path& marker, unsigned entry_offset);
}
