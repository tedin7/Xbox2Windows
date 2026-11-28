#pragma once

#include "gfx1013_compute_native.h"

// The result includes the emitted DXIL module and its exact desktop/guest binding layout.
namespace x2w {
size_t index_graphics_interfaces(const std::filesystem::path &source, const std::filesystem::path &cache);
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
// Without a manifest, reason (when given) receives a fixed refusal text: no register numbers, offsets or keys.
std::optional<std::string> format_compute_runtime_manifest(const NativeComputeShader &program,
                                                           std::string *reason = nullptr);
// rtv_request: the "rtv1 ..." text of an RTV variant, written as the manifest's last line for the runtime to match.
std::optional<std::string> format_graphics_runtime_manifest(const DirectGraphicsPair &pair,
                                                            std::string_view rtv_request = {});
}
