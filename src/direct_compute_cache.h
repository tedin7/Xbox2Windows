#pragma once

#include "gfx1013_compute_native.h"
#include <unordered_map>

// The result includes the emitted DXIL module and its exact desktop/guest binding layout.
namespace x2w {
enum class ComputePackResult { miss, prepared, unsupported };
struct ComputePackCounts { size_t prepared{}, unsupported{}; std::vector<std::string> errors; };
class ComputeFrontendPack {
    struct Record { std::string digest, payload; bool unsupported; };
    std::unordered_map<uint64_t, Record> records_;
    unsigned entry_{};
public:
    static ComputeFrontendPack load(const std::filesystem::path &file, unsigned entry);
    ComputePackResult lookup(const ComputeRequest &request, unsigned entry, NativeComputeShader &program,
                             std::string &diagnostic) const;
    size_t size() const { return records_.size(); }
};
ComputePackCounts build_compute_frontend_pack(const std::filesystem::path &requests, const std::filesystem::path &cache,
                                              unsigned entry);
size_t index_graphics_interfaces(const std::filesystem::path &source, const std::filesystem::path &cache);
// A logical match identifies the source shader. Only exact matches establish
// the compiled entry ABI, instruction offsets, and resource register layout.
GuestShaderStage reviewed_graphics_stage(std::string_view raw, GpuStage kind,
                                        const std::filesystem::path &cache, bool &exact);
// Signature component (f32/u32/i32) for supported typed IA and RTV formats.
unsigned dxil_io_component(unsigned dxgi_format);
NativeComputeShader prepare_direct_compute(const std::filesystem::path &input, unsigned entry, bool container,
                                           const std::filesystem::path &cache, bool &cache_hit,
                                           HeapViewContract heap_view = HeapViewContract::none);
DirectGraphicsPair prepare_direct_graphics(std::string_view a, std::string_view b,
                                          const std::filesystem::path &cache, bool &cache_hit,
                                          std::string_view root = {}, std::string_view input_layout = {},
                                          std::string_view pixel_entry = {}, bool write_cache = true,
                                          const std::vector<unsigned> &rtv_formats = {});
std::optional<std::string> format_compute_runtime_manifest(const NativeComputeShader &program);
// rtv_request: the "rtv1 ..." text of an RTV variant, written as the manifest's last line for the runtime to match.
std::optional<std::string> format_graphics_runtime_manifest(const DirectGraphicsPair &pair,
                                                            std::string_view rtv_request = {});
}
