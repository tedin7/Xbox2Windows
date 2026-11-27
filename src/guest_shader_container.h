#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace x2w {
enum class GpuStage { ls, hs, es, gs, vs, ps, ms, cs };
struct ShaderRootInput {
    uint32_t type, offset_dwords, sgpr_count, base_register;
    uint8_t user_data, root_parameter;
};
struct ShaderResourceUse {
    uint8_t type, root_type;
    uint32_t offset_dwords, lower_bound, stride;
    uint8_t dwords, user_data, root_parameter;
};
struct ShaderAttribute {
    uint8_t usage, row, mask, location, second_row, flags;
    uint8_t last_component{};
};
struct PixelExportPatch {
    unsigned target, word_offset;
    std::string full_precision;
};
struct ShaderSemantic {
    std::string name;
    unsigned index, system, component, row;
    uint8_t mask, io_mask;
};
struct GuestShaderStage {
    GpuStage stage;
    bool wave32;
    bool fixed_vertex_abi{};
    unsigned vgprs, lds_bytes;
    std::string code;
    std::vector<ShaderRootInput> inputs;
    std::vector<ShaderResourceUse> resources;
    std::array<unsigned, 3> threads{};
    unsigned rsrc2{};
    std::vector<ShaderAttribute> attributes, outputs;
    std::vector<PixelExportPatch> exports;
    std::vector<ShaderSemantic> input_signature, output_signature;
    unsigned pixel_input_address{}, pixel_input_enable = ~0u, depth_control{};
    // The signature component (9 f32, 5 u32, 4 i32) an RTV variant request asks of each color target; 0 when none.
    std::array<unsigned char, 8> target_components{};
    unsigned mesh_clip_control{}, max_vertices{}, max_primitives{}, topology{};
    std::string full_precision_code() const;
};
std::vector<GuestShaderStage> read_guest_shader_container(std::string_view container);
} // namespace x2w
