#pragma once
#include <d3d12.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <wrl/client.h>

void check(HRESULT hr, const char* stage);
// CheckFeatureSupport, routed through ID3D12Device16::CheckFeatureSupport1 with the app-local Agility
// SDK version when one is configured. The debug layer reports D3D12 989 for the unversioned call there.
HRESULT check_feature(ID3D12Device* device, D3D12_FEATURE feature, void* data, UINT size);
Microsoft::WRL::ComPtr<ID3D12Resource> resource(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc,
    D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear = nullptr);
void check_compute(ID3D12Device* device, bool tables);
void check_sampler_views(ID3D12Device* device, const char* shader_path);
bool check_startup_sampler_x(ID3D12Device* device, bool swapped);
int check_startup_cb_blend(ID3D12Device* device, bool swapped);
int check_startup_export_write_mask(ID3D12Device* device, bool unmasked);
int check_startup_composite_provenance(ID3D12Device* device);
int check_startup_range_provenance(ID3D12Device* device);
void check_dynamic_sampler_index(ID3D12Device* device);
// Returns 0 on the expected readback, 1 otherwise, 77 when R32_FLOAT lacks a needed capability.
int check_startup_sampler_lod(ID3D12Device* device, bool drop_clamp);
float check_startup_sampler_heap(ID3D12Device* device, const char* vertex_path, const char* pixel_path,
                                 std::uint32_t byte_offset, const char* manifest, const char* binding_case);
void check_startup_image_3d(ID3D12Device* device, const char* vertex_path, const char* pixel_path, float z,
                            float color[4], const char* sampler_line, bool array = false, UINT levels = 1,
                            bool one_dimensional = false, bool two_dimensional = false,
                            UINT first_mip = 0, UINT view_levels = 0);
void check_startup_image_load_d16(ID3D12Device* device, const char* vertex_path, const char* pixel_path,
                                  bool depth16, std::uint32_t words[64], bool array, bool sampled,
                                  bool half_float = false, bool red_only = false);
int check_startup_image_store(ID3D12Device* device, const char* vertex_path, const char* pixel_path, UINT x, UINT y,
                              const std::uint32_t expected[4]);
int check_startup_pixel_formatted_store(ID3D12Device* device, const char* vertex_path, const char* pixel_path,
                                        const char* sampler_line, const char* fixture_path);
void check_startup_flat_param(ID3D12Device* device, const char* vertex_path, const char* pixel_path,
                              std::uint32_t* words, UINT width = 1, UINT height = 1,
                              D3D12_CULL_MODE cull = D3D12_CULL_MODE_NONE, BOOL front_ccw = FALSE,
                              UINT vertices = 3, const std::uint32_t* constants = nullptr,
                              UINT constant_count = 4, const std::vector<std::uint32_t>& raw_srv = {});
void check_startup_entry_words(ID3D12Device* device, const char* vertex_path, const char* pixel_path,
                               const std::uint32_t* entry, UINT entry_count, std::uint32_t* words,
                               UINT width = 1, UINT height = 1, bool dynamic_heap = false);
void dump_raw_uav(ID3D12Device* device, const char* shader_path, const char* groups,
                  const char* bytes, const char* out_path);
void check_spirv_window(ID3D12Device* device, const char* shader_path);
void check_spirv_compaction(ID3D12Device* device, const char* shader_path);
void check_spirv_compare_sampler(ID3D12Device* device, const char* shader_path, const char* ordinary_register,
                                 const char* reference_path, const char* descriptors = nullptr);
void check_compute_file(ID3D12Device* device, const char* path, bool conversion_pair, const char* reference = nullptr,
                        bool dpp_case = false, bool sized_views = false, UINT uav_words = 0);
struct ComputeReadbackProgram {
    Microsoft::WRL::ComPtr<ID3D12RootSignature> signature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline;
};
ComputeReadbackProgram load_compute_store_program(ID3D12Device* device, const char* shader);
void check_compute_store_reference(ID3D12Device* device, const ComputeReadbackProgram& program, const char* reference);
void check_compute_texture_file(ID3D12Device* device, const char* path, const char* reference,
                                const char* sampler_manifest = nullptr);
bool check_compute_texture_msaa_file(ID3D12Device* device, const char* path, const char* reference,
                                     const char* vs_path, const char* ps_path);
bool check_compute_texture_store_file(ID3D12Device* device, const char* path, const char* reference);
void check_ffx_device_init(ID3D12Device* device);
void check_ffx_recurrent(ID3D12Device* device);
void check_dual_uav(ID3D12Device* device);
enum class DescriptorWindowCase { direct, stale_window, copied_record, stale_record, unknown_record };
void check_stream_root_srv(ID3D12Device* device);
void check_descriptor_window_submissions(ID3D12Device* device, DescriptorWindowCase scenario);
void check_cpu_srv_refresh(ID3D12Device* device, bool stale);
enum class MeshConstantBinding { root, root_pair, swapped_root_pair, table, swapped_table };
// multi_primitive: one native color draw whose export trace size comes from the fixture.
enum class MeshDrawMode { color, textured, depth_pair, multi_primitive };
struct MeshTextureProducers {
    const char* first;
    const char* second;
    bool bind_first;
};
bool check_mesh_entry_draw(ID3D12Device*, const char*, const char*, const char*, bool,
                           MeshConstantBinding = MeshConstantBinding::root,
                           MeshDrawMode = MeshDrawMode::color,
                           const MeshTextureProducers* = nullptr,
                           const char* amplification = nullptr);
bool check_mesh_amplified_draw(ID3D12Device*, const char* amplification, const char* mesh,
                               const char* pixel, const char* fixture);
bool check_mesh_shader_pipeline(ID3D12Device* device, bool force_fallback);
bool check_mesh_heap_sampler_draw(ID3D12Device*, const char* mesh, const char* pixel, const char* fixture);
void check_mesh_fallback_draw(ID3D12Device* device, const char* compute, const char* vertex,
                              const char* pixel, const char* fixture, MeshDrawMode mode = MeshDrawMode::color,
                              const MeshTextureProducers* = nullptr);
void check_texture_mips(ID3D12Device* device);
void check_depth_mask_draw(ID3D12Device*, const char*, const char*, const char*, bool, bool color = false);
void check_sampled_coverage(ID3D12Device*, const char*, const char*);
void check_dual_pixel(ID3D12Device*, const char*, const char*, bool);
struct ReadbackMismatch : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct PixelMismatch : ReadbackMismatch {
    PixelMismatch() : ReadbackMismatch("AOT color export differs from CPU reference") {}
};

void check_translated_pixel(ID3D12Device* device, const char* path, bool swapRB, const char* reference = nullptr,
                            UINT8 writeMask = D3D12_COLOR_WRITE_ENABLE_ALL, const char* rawReadback = nullptr);
void check_attribute_pixel(ID3D12Device* device, const char* shader, const char* reference, bool rg);
void check_compute_to_pixel(ID3D12Device* device, const char* compute, const char* pixel,
                            const char* reference, const char* previous);
void check_pixel_quad_lod(ID3D12Device* device, const char* shader, const char* rawReadback = nullptr,
                         bool allowUpperRawLodStep = false);
void check_pixel_mip_clamp(ID3D12Device* device, const char* shader, unsigned expectedLevel);
void check_pixel_mip_blend(ID3D12Device* device, const char* shader, float clamp, float tolerance);
void check_pixel_compare(ID3D12Device* device, const char* shader, const char* reference, bool twoMips = false);
void check_pixel_gather(ID3D12Device* device, const char* shader, const char* reference);
void check_pixel_compare_gather(ID3D12Device* device, const char* shader, const char* reference,
                                const char* sampler_line, const char* vsPath = nullptr);
bool check_mesh_shadow_draw(ID3D12Device*, const char* mesh, const char* receiver, const char* caster,
                            const char* fixture, bool stale);
void check_bc7_pixel(ID3D12Device* device, const char* shader, const char* input,
                     const char* capture, const char* mip1 = nullptr);
void check_bc7_tail(ID3D12Device* device, const char* shader, const char* firstMip, int count, char** paths);
void check_bvh_root_roundtrip(ID3D12Device* device);
void check_buffer_ops(ID3D12Device* device);
void check_bvh_reads(ID3D12Device* device);
void check_bvh_atomics(ID3D12Device* device);
void check_bvh_numeric(ID3D12Device* device);
void check_bvh_packed(ID3D12Device* device);
void check_bvh_hits(ID3D12Device* device);
bool check_bvh_differential(ID3D12Device* device, bool forcePipeline);
bool check_bvh_waves(ID3D12Device* device, bool force64);
bool check_gpu_alias(ID3D12Device* device);
bool check_gpu_alias_window(ID3D12Device* device);
bool check_gpu_alias_lookup(ID3D12Device* device);
bool check_gpu_alias_lookup_writes(ID3D12Device* device);
bool check_gpu_alias_atomic_waterfall(ID3D12Device* device);
bool check_gpu_alias_registry(ID3D12Device* device);
bool check_wave_shuffle(ID3D12Device* device);
void check_wave64_emulate(ID3D12Device* device);
bool check_shader_wave(ID3D12Device* device, unsigned forced);
bool check_dxr_address_ranges(ID3D12Device* device);
bool check_dxr_placed_build(ID3D12Device* device);
bool check_dxr_slot70_as(ID3D12Device* device);
bool check_dxr_rays(ID3D12Device* device, bool usePipeline);
bool check_dxr_update(ID3D12Device* device, bool usePipeline);
bool check_dxr_facing(ID3D12Device* device, bool usePipeline);
bool check_dxr_opacity(ID3D12Device* device, bool warp, bool usePipeline, float nearT);

Microsoft::WRL::ComPtr<ID3D12Resource> make_buffer(ID3D12Device* device, UINT64 length, D3D12_HEAP_TYPE heap,
    D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE);
void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* buffer,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);

struct ComputeCommands {
    ComputeCommands(ID3D12Device* device, ID3D12PipelineState* pipeline,
                    D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_COMPUTE);
    // Records on a new list for an existing queue of the list's type. Releasing the commands then
    // leaves that queue alive.
    ComputeCommands(ID3D12Device* device, ID3D12CommandQueue* existing, ID3D12PipelineState* pipeline);
    void submit_and_wait(ID3D12Device* device);
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
};
