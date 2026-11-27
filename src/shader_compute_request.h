#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace x2w {
struct ComputeRequest {
    std::string code, data, root, description, sa, sb;
    uint64_t key;
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
void validate_compute_request_shape(const ComputeRequest &request);
} // namespace x2w
