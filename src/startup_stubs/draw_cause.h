#pragma once

namespace x2w_startup {

inline constexpr const char* const not_drawn_names[] = {
    "state_not_ready",
    "translation_pending",
    "missing_guest_root",
    "live_compute_pipeline",
    "writable_texture_indirect",
    "unsupported_resource_or_entry",
    "read_only_depth_write",
    "pipeline_or_sampler_heap_unavailable",
    "pixel_uav_unresolved",
    "sampled_texture_mismatch",
    "sampled_texture_unbindable",
    "incomplete_vertex_or_table_buffer",
    "formatted_buffer_too_large",
    "formatted_buffer_unresolved",
    "indexed_validation_failed",
    "translation_refused",
    "translation_failed_to_load",
    "d16_view_not_ready",
    "formatted_buffer_not_ready",
    "buffer_copy_failed",
    "index_data_unavailable",
};
constexpr int not_drawn_sites = sizeof(not_drawn_names) / sizeof(not_drawn_names[0]);
constexpr int translation_pending_site = 1;
constexpr int translation_refused_site = not_drawn_sites - 6;
constexpr int translation_failed_to_load_site = not_drawn_sites - 5;
constexpr int d16_view_not_ready_site = not_drawn_sites - 4;
constexpr int formatted_buffer_not_ready_site = not_drawn_sites - 3;
constexpr int buffer_copy_failed_site = not_drawn_sites - 2;
constexpr int index_data_unavailable_site = not_drawn_sites - 1;

}  // namespace x2w_startup
