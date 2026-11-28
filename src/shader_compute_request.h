#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace x2w {
uint64_t fnv1a_64(std::string_view bytes);
uint64_t graphics_request_key(std::string_view a, std::string_view b, std::string_view root,
                              std::string_view input_layout = {}, std::string_view pixel_entry = {},
                              std::string_view revision = {});
uint64_t compute_request_key(std::string_view code, std::string_view data, std::string_view root,
                             std::string_view description, std::string_view sa, std::string_view sb,
                             std::string_view revision = {});
uint64_t mesh57_request_key(std::string_view code, std::string_view ps57, std::string_view root);

struct PixelInputDescription {
    uint32_t ena{};
    uint32_t addr{};
    std::string entry;
    std::string canonical_text;
};
PixelInputDescription parse_pixel_input_description(std::string_view text, uint64_t expected_key);
std::string format_pixel_input_description(uint64_t key, uint32_t ena, uint32_t addr);

struct RtvVariantDescription {
    uint64_t base_key{};
    std::vector<unsigned> formats;
};
RtvVariantDescription parse_rtv_variant_text(std::string_view text, uint64_t expected_key);
std::string format_rtv_variant_text(uint64_t base_key, const std::vector<unsigned> &formats);

struct ComputeRequest {
    std::string code, data, root, description, sa, sb;
    uint64_t key;
    bool wave32 = true;
};
struct ComputeContainer {
    std::string bytes, root;
};
struct GraphicsRequest {
    std::string a, b, root, input_layout, pixel_entry, pixel_input_description;
    uint64_t key{};
    // An RTV variant (req_<key>.rtv): its "rtv1 <base key> <format> ..." text and the ordered target formats. The
    // parts above are then its base request's, checked against the base key, and key is the variant's.
    std::string rtv_request;
    std::vector<unsigned> rtv_formats;
};
// marker: req_<key>.a, or req_<key>.rtv for an RTV variant of a base request in the same directory.
GraphicsRequest read_graphics_request(const std::filesystem::path &marker);
std::string read_shader_container_bytes(const std::filesystem::path &file);
ComputeContainer read_compute_container(const std::filesystem::path &file);
// Verify the content key over all six immutable capture parts before exposing any bytes.
ComputeRequest read_compute_request(const std::filesystem::path &marker);
} // namespace x2w
