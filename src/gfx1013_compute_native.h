#pragma once
#include "mesh_pixel_heap.h"
#include "shader_compute_request.h"
#include <array>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

struct D3D12_ROOT_SIGNATURE_DESC1;
struct D3D12_ROOT_DESCRIPTOR_TABLE1;
namespace x2w {
enum class GpuStage;
struct GuestShaderStage;
class UnsupportedComputeShader : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};
enum class ComputeViewKind {
    buffer_srv,
    typed_buffer_srv,
    texture_srv,
    texture_uav,
    raw_vector_srv,
    half_scalar_uav,
    heap_stride_srv,
    raw_buffer_uav,
    typed_buffer_uav,
    half_buffer_uav,
    root_buffer_srv,
    comparison_texture_srv,
    half2_texture_uav,
    half4_texture_uav,
    cube_sample_srv,
    cube_comparison_srv,
    indexed_atomic_uav,
    d16_sample_srv,
    byte_window_srv,
<<<<<<< HEAD
    heap_buffer_srv, // A raw buffer read through ResourceDescriptorHeap[slot]; the slot is a GPU value.
    heap_typed_srv, // A one-channel 32-bit typed buffer read the same way (HeapViewContract::typed_dword).
    // The directly indexed sampler heap; entry n = title byte offset 16 * n (index parity); the runtime fills entries with
    // dynamic_sampler() (LOD clamp applied, valid for the explicit-LOD forms translated here); entry 2047 is the runtime
    // fallback and never read.
    heap_sampler
||||||| ca6593968
    heap_buffer_srv // A raw buffer read through ResourceDescriptorHeap[slot]; the slot is a GPU value.
=======
    heap_buffer_srv, // A raw buffer read through ResourceDescriptorHeap[slot]; the slot is a GPU value.
    packed_d16_x_uav, // Single low-halfword payload; desktop buffer view determines the element size.
    indexed_raw_srv // Fixed-stride record read. required_bytes is one complete record and its checked stride.
>>>>>>> a48d39dc1
};
inline bool accepts_cbv_window(ComputeViewKind kind) {
    return kind == ComputeViewKind::buffer_srv || kind == ComputeViewKind::byte_window_srv;
}
inline constexpr unsigned compute_resource_heap = ~0u;
// the directly indexed sampler heap; entry n is title byte offset 16 * n
inline constexpr unsigned compute_sampler_heap = ~0u - 1;
// A binding of the whole descriptor heap: it has no table register and no root parameter.
inline bool compute_heap_binding(ComputeViewKind kind) {
    return kind == ComputeViewKind::heap_buffer_srv || kind == ComputeViewKind::heap_typed_srv;
}
inline bool compute_buffer_uav(ComputeViewKind kind) {
    return kind == ComputeViewKind::raw_buffer_uav || kind == ComputeViewKind::typed_buffer_uav ||
           kind == ComputeViewKind::half_buffer_uav || kind == ComputeViewKind::packed_d16_x_uav ||
           kind == ComputeViewKind::indexed_atomic_uav;
}
struct IndexedAtomicView {
    unsigned stride = 0, count = 0; // Bytes per guest index and number of valid indices.
};
inline bool compute_uav(ComputeViewKind kind) {
    return kind == ComputeViewKind::texture_uav || kind == ComputeViewKind::half_scalar_uav ||
           kind == ComputeViewKind::half2_texture_uav || kind == ComputeViewKind::half4_texture_uav || compute_buffer_uav(kind);
}
struct ComputeViewBinding {
    ComputeViewKind kind;
    unsigned root_parameter;
    unsigned byte_offset;
    unsigned shader_register;
    unsigned required_bytes;
    unsigned array_count = 1, array_position = 0;
    unsigned image_dimension = 1; // ISA dimension field; 1 denotes Texture2D.
    enum class ScalarTable { unknown, reflected_cbv, declared_cbv, srv };
    ScalarTable scalar_table = ScalarTable::unknown;
};
bool apply_reflected_table_use(ComputeViewBinding &binding, const GuestShaderStage &stage,
                               std::optional<unsigned> packed_sgpr = {});
void apply_scalar_table_layout(ComputeViewBinding &binding, const D3D12_ROOT_DESCRIPTOR_TABLE1 &table);
struct ComputeSampler {
    std::array<unsigned, 4> words;
    bool explicit_lod;
    bool comparison = false;
};
struct ComputeConstants {
    unsigned root_parameter;
    unsigned first_sgpr;
    unsigned count;
};
// Both graphics stages read a single desktop b0. Pack each guest root once,
// in parameter order, irrespective of the SGPR order in either stage.
struct GraphicsConstantLayout {
    std::array<unsigned, 16> counts{};
    unsigned offset(unsigned parameter) const {
        unsigned position = 0;
        for (unsigned r = 0; r < parameter; ++r)
            position += counts.at(r);
        return position;
    }
    void include(unsigned parameter, unsigned count) {
        if (parameter >= counts.size() || !count || count > 32)
            throw UnsupportedComputeShader("invalid graphics root constant range");
        if (counts[parameter] < count)
            counts[parameter] = count;
        if (offset(static_cast<unsigned>(counts.size())) > 32)
            throw UnsupportedComputeShader("linked graphics root exceeds 32 words");
    }
};
struct GraphicsLinkage {
    std::string name;
    unsigned index, row, first, width, component, used, kind, interpolation;
};
struct MeshHeapRead {
    unsigned root_parameter, dword, mask; // R32_UINT SRV at (root constant & mask).
};
struct NativeComputeShader {
    std::vector<PixelHeapSpan> pixel_heap_spans;
    std::vector<MeshHeapRead> mesh_heap_reads;
    std::vector<GraphicsLinkage> inputs, outputs;
    std::vector<ComputeConstants> constants;
    std::string hlsl, dxil_ir;
    std::vector<ComputeViewBinding> bindings;
    std::vector<ComputeSampler> samplers;
    bool sampler_heap = false;
    unsigned instruction_count{};
    std::array<unsigned, 3> group{};
    bool vertex_fetch = false;
    std::array<unsigned, 8> pixel_writes{};
};
// The view class a GPU-selected descriptor heap slot is read as. none: the slot and its view type are not known, so a
// heap read is refused. raw_buffer: the capture says the selected slot is a raw buffer SRV. other: the capture names a
// view type that is not translated (refused when the shader reads the heap).
// Under raw_buffer the direct path also reads the byte-address rewrite of the selected descriptor (stride read,
// max(stride, 1), byte count, stride cleared, raw out-of-bounds mode) as the same raw view. The caller guarantees that
// every slot the program can select is a raw view whose size is the guest range num_records * max(stride, 1); the
// translator does not check that, nor the swizzle bits (word 1 bits 31:30) or the other word 3 fields, and a program
// with two heap loads from different selectors is outside what one captured selection covers.
// typed_dword: only typed loads (buffer_load_format_x, an idxen element index or none, no offsets) read the heap; raw
// loads and byte windows are refused. The caller guarantees that every slot the program can select is a typed buffer SRV
// of R32_UINT, R32_SINT or R32_FLOAT (default mapping, no flags, no structure stride), bound as R32_UINT over the same
// resource and element range, and that the guest descriptor it stands for has stride 4, a 32-bit UINT/SINT/FLOAT data
// format, identity dst_sel_x, swizzle and add_tid disabled, and the out-of-bounds rule "index >= num_records reads 0",
// with num_records equal to the view's element count. The translator checks none of these values.
enum class HeapViewContract : unsigned { none, raw_buffer, typed_dword, other };
NativeComputeShader translate_compute(const ComputeRequest &request, unsigned entry_byte_offset,
                                      bool direct_dxil = false);
NativeComputeShader translate_compute_container(ComputeContainer container,
                                                HeapViewContract heap_view = HeapViewContract::none);
struct NativeVertexElement {
    std::string semantic;
    unsigned index, format, slot, offset, classification, step_rate;
    bool used = false;
};
struct DirectGraphicsPair {
    NativeComputeShader vertex, pixel;
    std::vector<NativeVertexElement> input_layout;
    // GuestShaderStage::target_components of the pixel stage: the target types an RTV variant asked for.
    std::array<unsigned char, 8> target_components{};
    GraphicsConstantLayout constant_layout() const {
        GraphicsConstantLayout layout;
        for (auto *stage : {&vertex, &pixel})
            for (const auto &block : stage->constants)
                layout.include(block.root_parameter, block.count);
        return layout;
    }
};
DirectGraphicsPair translate_graphics_pair(std::string_view a, std::string_view b,
                                          const D3D12_ROOT_SIGNATURE_DESC1 &layout, std::string_view pixel_entry = {},
                                          std::optional<GpuStage> first_stage = {},
                                          const std::array<unsigned char, 8> &target_components = {},
                                          std::optional<std::array<bool,2>> stage_wave32 = {});
DirectGraphicsPair translate_graphics_pair(std::string_view a, std::string_view b);
void validate_raster_link(const NativeComputeShader &producer, const NativeComputeShader &pixel);
DirectGraphicsPair translate_reflected_graphics(GuestShaderStage &vs, GuestShaderStage &ps,
                                                const D3D12_ROOT_SIGNATURE_DESC1 &layout);
std::string native_rectangle_geometry(const std::vector<GraphicsLinkage>& vertex_outputs,
                                      const std::vector<GraphicsLinkage>& pixel_inputs);
NativeComputeShader translate_graphics_container(const std::string &bytes, bool pixel);
struct MeshRootInput {
    unsigned sgpr, root_parameter, constant_count; // Zero words denotes a table unless root_address is set.
    bool root_address{};
};
struct MeshStageLayout {
    unsigned threads, max_vertices, max_primitives, topology, group_stride;
    bool wave32;
    std::vector<MeshRootInput> root_inputs{};
    std::optional<unsigned> resource_heap_sgpr{};
    bool validate_uint_heap_reads{};
    bool packed_parameters{};
};
NativeComputeShader translate_mesh_pixel_program(std::string_view code, unsigned enable, unsigned address,
                                                 unsigned wave_size, const D3D12_ROOT_SIGNATURE_DESC1 *root = nullptr);
NativeComputeShader translate_mesh_program(const std::string &code, const MeshStageLayout &layout);
NativeComputeShader translate_mesh_program(const std::string &code, const MeshStageLayout &layout,
                                           const D3D12_ROOT_SIGNATURE_DESC1 &root);
NativeComputeShader translate_mesh_container(const std::string &container, bool validate_uint_heap_reads = false);
} // namespace x2w
