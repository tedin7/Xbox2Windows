#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "startup_stream_draw.h"
#include "startup_stream_setup.h"
#include "startup_slot67_indirect.h"

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) void x2w_d3d12_compute_census();
extern "C" __declspec(dllimport) std::size_t x2w_d3d12_not_drawn_keys_report(char* out, std::size_t capacity);
extern "C" __declspec(dllimport) UINT x2w_d3d12_fixture_fill_upload_ring(void*, UINT, BOOL);
extern "C" __declspec(dllimport) void x2w_d3d12_compute_refusal_stats(
    LONG64* holds, LONG64* repeated_reads, LONG64* background_reads, LONG* untranslated);

struct ComputeRefusalStats {
    LONG64 holds, repeated_reads, background_reads;
    LONG untranslated;
};

static ComputeRefusalStats compute_refusal_stats()
{
    ComputeRefusalStats stats{};
    x2w_d3d12_compute_refusal_stats(&stats.holds, &stats.repeated_reads, &stats.background_reads, &stats.untranslated);
    return stats;
}

// A slot-55 patch is a tagged union with room for the full eight-target blend state.
struct Patch { UINT tag; D3D12_BLEND_DESC blend; };
static_assert(sizeof(Patch) == 0x14c);

static bool derive(void* device, void*& pipeline, bool child)
{
    using Create = HRESULT (WINAPI*)(void*, void*, UINT, const Patch*, REFIID, void**);
    auto create = reinterpret_cast<Create>((*static_cast<void***>(device))[55]);
    Patch preserve{2}, write{2}, sample_mask{3};
    const UINT all_samples = ~0u;
    std::memcpy(&sample_mask.blend, &all_samples, sizeof(all_samples));
    auto& t = preserve.blend.RenderTarget[0];
    t = {TRUE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_BLEND_ZERO,
         D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
    write.blend = preserve.blend;
    write.blend.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    write.blend.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    void* next = nullptr;
    if (create(device, pipeline, 1, nullptr, __uuidof(ID3D12PipelineState), &next) != E_INVALIDARG || next) return false;
    if (FAILED(create(device, pipeline, 1, &preserve, __uuidof(ID3D12PipelineState), &next)) || !next) return false;
    pipeline = next;
    if (child) {
        Patch patches[] = {preserve, write, sample_mask};
        if (FAILED(create(device, pipeline, 3, patches, __uuidof(ID3D12PipelineState), &next)) || !next) return false;
        pipeline = next;
    }
    // A grandchild with only a non-blend patch keeps its nearest ancestor's state.
    if (FAILED(create(device, pipeline, 1, &sample_mask, __uuidof(ID3D12PipelineState), &next)) || !next) return false;
    pipeline = next;
    return true;
}

// A slot-57 description as the stand-in reads it: 64-bit words 0 (root signature), 1-2 (state block
// A and its size), 5-6 and 9-10 (the two shaders). spec lists "offset:word" pairs in hex, written
// into a zeroed state block A; the colour-blend words of the eight targets start at +0xD0.
// The shaders and the root signature are read from the files the stream pipeline was made from, so
// both pipelines share one translation.
static bool create_slot57(void* device, const char* vs_path, const char* ps_path, const char* root_path, const char* spec,
                          void*& pipeline)
{
    static std::vector<char> vs, ps;
    std::ifstream vs_file(vs_path, std::ios::binary), ps_file(ps_path, std::ios::binary), root_file(root_path, std::ios::binary);
    vs.assign(std::istreambuf_iterator<char>(vs_file), {});
    ps.assign(std::istreambuf_iterator<char>(ps_file), {});
    const std::vector<char> blob{std::istreambuf_iterator<char>(root_file), {}};
    if (vs.empty() || ps.empty() || blob.empty()) return false;
    using CreateRoot = HRESULT (WINAPI*)(void*, UINT, const void*, SIZE_T, REFIID, void**);
    void* root = nullptr;
    if (FAILED(reinterpret_cast<CreateRoot>((*static_cast<void***>(device))[16])(device, 0, blob.data(), blob.size(),
                                                                                  __uuidof(ID3D12RootSignature), &root))) return false;
    alignas(8) static unsigned char desc[0x300];
    alignas(8) static unsigned char state_a[0x200];
    for (const char* at = spec; *at;) {
        char* end = nullptr;
        const unsigned long offset = std::strtoul(at, &end, 16);
        if (*end != ':' || offset % 4 || offset + 4 > sizeof(state_a)) return false;
        const UINT word = static_cast<UINT>(std::strtoul(end + 1, &end, 16));
        std::memcpy(state_a + offset, &word, sizeof(word));
        if (*end && *end != ',') return false;
        at = *end ? end + 1 : end;
    }
    const UINT64 words[12] = {reinterpret_cast<UINT64>(root), reinterpret_cast<UINT64>(state_a), sizeof(state_a), 0, 0,
                              reinterpret_cast<UINT64>(vs.data()), vs.size(), 0, 0, reinterpret_cast<UINT64>(ps.data()), ps.size(), 0};
    std::memcpy(desc, words, sizeof(words));
    using Create = HRESULT (WINAPI*)(void*, const void*, REFIID, void**);
    pipeline = nullptr;
    return SUCCEEDED(reinterpret_cast<Create>((*static_cast<void***>(device))[57])(device, desc, __uuidof(ID3D12PipelineState), &pipeline)) &&
           pipeline;
}

// Binds the pipeline on a guest list and records `calls` ExecuteIndirect calls (list slot 61) with a Draw or a
// Dispatch signature. No root or target is bound, so every call is refused and nothing is drawn.
static bool bind_and_execute_indirect(void* device, void* pipeline, bool dispatch, int calls, bool direct_draw = false)
{
    auto* guest = static_cast<ID3D12Device*>(device);
    D3D12_INDIRECT_ARGUMENT_DESC argument{};
    argument.Type = dispatch ? D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH : D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
    const D3D12_COMMAND_SIGNATURE_DESC signature_desc{
        dispatch ? UINT{sizeof(D3D12_DISPATCH_ARGUMENTS)} : UINT{sizeof(D3D12_DRAW_ARGUMENTS)}, 1, &argument, 0};
    ComPtr<ID3D12CommandSignature> signature;
    ComPtr<ID3D12CommandAllocator> allocator;
    if (FAILED(guest->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature))) ||
        FAILED(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)))) return false;
    const auto slots = [](void* object) { return *static_cast<void***>(object); };
    using CreateList = HRESULT (WINAPI*)(void*, const void*, void*, void*, REFIID, void**);
    const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    void* list = nullptr;
    if (FAILED(reinterpret_cast<CreateList>(slots(device)[73])(device, list_desc, allocator.Get(), nullptr,
                                                                 __uuidof(ID3D12GraphicsCommandList), &list)) || !list) return false;
    using Reset = HRESULT (WINAPI*)(void*, void*, void*, void*);
    using Bind = void (WINAPI*)(void*, void*);
    using Execute = void (WINAPI*)(void*, void*, UINT, void*, UINT64, void*, UINT64);
    using Close = HRESULT (WINAPI*)(void*);
    if (FAILED(reinterpret_cast<Reset>(slots(list)[139])(list, allocator.Get(), nullptr, nullptr))) return false;
    reinterpret_cast<Bind>(slots(list)[25])(list, pipeline);
    if (direct_draw) {
        // A plain DrawInstanced with no render target bound: refused before any translation lookup.
        using Draw = void (WINAPI*)(void*, UINT, UINT, UINT, UINT);
        for (int i = 0; i < calls; ++i) reinterpret_cast<Draw>(slots(list)[12])(list, 3, 1, 0, 0);
    } else {
        for (int i = 0; i < calls; ++i) reinterpret_cast<Execute>(slots(list)[61])(list, signature.Get(), 1, nullptr, 0, nullptr, 0);
    }
    const bool closed = SUCCEEDED(reinterpret_cast<Close>(slots(list)[9])(list));
    static_cast<IUnknown*>(list)->Release();
    return closed;
}

// A slot-86 pipeline with desktop shaders that use no resources, over the stream root: a desktop pipeline.
static void* desktop_pipeline(void* device)
{
    static const char vs_text[] = "float4 main(uint id : SV_VertexID) : SV_Position { return float4(0, 0, 0, 1); }";
    static const char ps_text[] = "float4 main() : SV_Target { return float4(1, 0, 0, 1); }";
    ComPtr<ID3DBlob> vs, ps;
    if (FAILED(D3DCompile(vs_text, sizeof(vs_text) - 1, nullptr, nullptr, nullptr, "main", "vs_5_0", 0, 0, &vs, nullptr)) ||
        FAILED(D3DCompile(ps_text, sizeof(ps_text) - 1, nullptr, nullptr, nullptr, "main", "ps_5_0", 0, 0, &ps, nullptr)))
        return nullptr;
    struct alignas(void*) Root { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; void* value; };
    struct alignas(void*) Code { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; D3D12_SHADER_BYTECODE value; };
    struct alignas(void*) Formats { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; D3D12_RT_FORMAT_ARRAY value; };
    struct alignas(void*) Topology { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; D3D12_PRIMITIVE_TOPOLOGY_TYPE value; };
    struct { Root root; Code vs, ps; Formats targets; Topology topology; } stream{
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, stream_root_signature()},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, {vs->GetBufferPointer(), vs->GetBufferSize()}},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, {ps->GetBufferPointer(), ps->GetBufferSize()}},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, {{DXGI_FORMAT_R8G8B8A8_UNORM}, 1}},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE}};
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
    using Create = HRESULT (WINAPI*)(void*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
    void* pipeline = nullptr;
    return SUCCEEDED(reinterpret_cast<Create>((*static_cast<void***>(device))[86])(device, &desc, __uuidof(ID3D12PipelineState),
                                                                                   &pipeline)) ? pipeline : nullptr;
}

// A slot-67 compute description as the stand-in reads it: 64-bit words 0 (root signature), 1-2 (state
// block A), 3-4 (state block B: 20 fixed bytes, dwords 0x1234abcd, 0xffff0000, 0, 0, 0), 5-6 (program code), 7-8 (data blob), 9-10 (pointers into this
// process: heap and stack) and 11 (a plain value). X2W_TEST_COMPUTE_ROOT=<file> creates the root signature
// from that serialized blob instead of using the stream root.
//
// Cases (each pipeline is a placeholder):
//   valid            the description twice
//   order            valid, then refused (no root), then valid with another data blob
//   code-limit       code of exactly 1 MiB (the program, then zeros)    code-over: 1 MiB + 1
//   state-limit      state block A of exactly 4 KiB                     state-over: 4 KiB + 1
//   data-limit       data blob of exactly 1 MiB                         data-over: 1 MiB + 1
//   stateb-limit     state block B of exactly 4 KiB                     stateb-over: 4 KiB + 1
//   data-pointer     valid, with the first data qword pointing at state block A
//   code-zero, data-zero, no-root, code-unreadable, code-past-end, data-unreadable, state-unreadable,
//   stateb-unreadable, desc-unreadable
//   dispatch, dispatch-swapped, dispatch-none, dispatch-sampled, dispatch-sampled-unbound, dispatch-sampled-2x1, dispatch-heap,
//   dispatch-heap-outside, dispatch-heap-word*, dispatch-input-*: one creation, then dispatch_slot67 below
//   texture-store, texture-store-none, texture-store-denied, texture-store-occupied, texture-store-copied: one creation, then texture_slot67 below
// The unreadable cases point at a PAGE_NOACCESS page; code-past-end ends the code on the last byte before one.
static void* slot67_pipeline;
static void* slot67_root;
static void* slot67_other_root;  // a second root signature object, for the root-change controls

// A console resource through device slot 127 (CreateCommittedResource3).
static void* make_committed(void* device, D3D12_HEAP_TYPE type, const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES state)
{
    using Create = HRESULT(WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, UINT, const D3D12_RESOURCE_DESC*, UINT, const void*,
                                    void*, UINT, const void*, const GUID*, void**);
    const D3D12_HEAP_PROPERTIES heap{type};
    void* resource = nullptr;
    const HRESULT hr = reinterpret_cast<Create>((*static_cast<void***>(device))[127])(
        device, &heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, nullptr, 0, nullptr, &__uuidof(ID3D12Resource), &resource);
    return SUCCEEDED(hr) ? resource : nullptr;
}

// Case A of startup_gfx2hlsl_compute_check.py through the stand-in's own list: two root constants (word 0 by slot
// 37, word 1 by slot 39; a slot 40 call with other words in between must not reach them), a
// table (slot 35) of two 16-byte hardware buffer descriptors written into the title heap's memory, and Dispatch
// (slot 14) of 3x2x2 groups with the slot 67 placeholder bound. Input and output are placed buffers 64 KiB apart
// in one allocation, so both alias this process's memory and the result is read from that memory after the queue's
// fence. dispatch-swapped exchanges the two descriptors (the output then names the buffer without unordered
// access); dispatch-none records everything but the Dispatch. The 320 output words go to X2W_TEST_DISPATCH_OUT.
// dispatch-sampled adds what the sampling variant of the case reads: a committed 1x1 RGBA8 texture filled by
// CopyTextureRegion (list slot 16) and a shader resource view of it (device slot 18) whose token the stand-in
// writes at table byte 64. dispatch-sampled-unbound creates no view, so that entry holds no token.
// dispatch-sampled-2x1 makes the texture 2x1 with red 0 in its left texel and 1.0 in its right one.
// dispatch-heap runs the program of startup_gfx2hlsl_heap_record_check.py: a third root constant, 2, indexes the
// 32-byte heap record that holds the input descriptor (heap byte 64). Heap bytes 0 and 32 hold the output's
// descriptor, so an index that is ignored or scaled by 16 names a buffer the dispatch refuses as an input.
// dispatch-heap-outside sets the constant to 0x07ffffff, whose record is past any heap.
// dispatch-heap-format-* dispatches one four-thread group and binds a typed view or native descriptor at record 2.
// dispatch-heap-format-indexed takes record 2 from a literal byte offset in the uploaded index buffer.
// dispatch-heap-word runs the program of startup_gfx2hlsl_heap_word_check.py: the constant is 1 and picks the word
// of the upload buffer at table byte 96 that holds the index, 2. dispatch-heap-word-outside makes that word
// 0x07ffffff, dispatch-heap-word-null leaves the descriptor of that buffer null (the word reads 0, record 0 is the input),
// dispatch-heap-word-past ends the buffer at that word, and dispatch-heap-word-placed names the placed
// input buffer instead of upload memory.
// Controls on the recorded state: dispatch-missing-word leaves out the slot 37 call; dispatch-root-changed sets
// another compute root and the first one again after the constants, dispatch-same-root sets the same root again;
// dispatch-reset closes and resets the list after the constants; dispatch-wrong-root records under the other
// root; dispatch-same-buffer names the output buffer as the input too.
// Input descriptors the dispatch refuses, one reason each: dispatch-input-empty has zero records (a nonzero base,
// word 3 zero), dispatch-input-past ends 16 bytes past the input buffer, dispatch-input-unplaced names memory between
// the two buffers, dispatch-input-copied a third buffer placed there at an offset the desktop cannot alias, and
// dispatch-input-null (all four words zero, an unbound buffer) is not refused and reads zeros;
// dispatch-input-token writes the token of a texture view where the input descriptor was.
// For any mode, X2W_TEST_DISPATCH_IN names 1280 bytes that replace the input words, and X2W_TEST_DISPATCH_FIELDS
// names six words that replace, for the input and then the output descriptor, the stride (word 1 bits 16-31), the
// record count (word 2) and word 3, which are otherwise 4, 320 and 0.
static int dispatch_slot67(IDXGIAdapter* adapter, void* device, const std::string& mode)
{
    constexpr UINT words = 320;
    const bool title_input = mode.rfind("dispatch-input-title-", 0) == 0;
    const bool view_tokens = mode.rfind("dispatch-input-view-tokens", 0) == 0;
    const bool gpu_input = mode == "dispatch-input-title-gpu";
    const bool alias_input = mode == "dispatch-input-title-alias";
    const bool large_input = mode == "dispatch-input-title-large" || gpu_input || alias_input;
    constexpr UINT gpu_offset = 16;
    const bool ring_input = mode == "dispatch-input-title-ring" || mode == "dispatch-input-title-ring-full";
    const bool protected_input = mode == "dispatch-input-title-protected";
    const UINT input_bytes = large_input ? 40u << 20 : 12u << 10;
    const UINT record_offsets[] = {0, input_bytes / 2, input_bytes - 16};
    const bool indirect = mode.rfind("dispatch-indirect-", 0) == 0;
    const UINT groups = mode == "dispatch-indirect-zero" ? 0u : mode == "dispatch-indirect-2" ? 2u : 5u;
    auto* memory = static_cast<UINT*>(VirtualAlloc(nullptr, indirect ? 0x30000 : 0x20000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!memory) return 70;
    UINT* const buffers[2] = {memory, memory + 0x10000 / sizeof(UINT)};
    for (UINT i = 0; i < words; ++i) {
        buffers[0][i] = 0x8b02a7f1u + i * 0x413b1u;
        buffers[1][i] = 0xc5c5c5c5u;
    }
    UINT fields[2][3] = {{4, words, 0}, {4, words, 0}};
    const auto from_file = [](const char* variable, void* data, std::streamsize bytes) {
        char path[MAX_PATH];
        const DWORD n = GetEnvironmentVariableA(variable, path, MAX_PATH);
        return !n || (n < MAX_PATH && std::ifstream(path, std::ios::binary).read(static_cast<char*>(data), bytes).gcount() == bytes);
    };
    if (!from_file("X2W_TEST_DISPATCH_IN", buffers[0], words * sizeof(UINT)) ||
        !from_file("X2W_TEST_DISPATCH_FIELDS", fields, sizeof(fields))) return 2;
    UINT input[words];
    std::memcpy(input, buffers[0], sizeof(input));
    const auto word = [](const void* value) { return reinterpret_cast<StreamWord>(value); };
    void* placed[2]{};
    UINT descriptors[2][4]{};
    for (int i = 0; i < 2; ++i) {
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, words * sizeof(UINT), 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR, i ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE};
        if (FAILED(static_cast<HRESULT>(invoke(device, 133, word(buffers[i]), word(&desc), D3D12_RESOURCE_STATE_COMMON, 0,
                word(&__uuidof(ID3D12Resource)), word(&placed[i]))))) return 71;
        const UINT64 address = reinterpret_cast<UINT64>(buffers[mode == "dispatch-same-buffer" ? 1 : i]);
        const UINT view[4] = {static_cast<UINT>(address), static_cast<UINT>(address >> 32) | fields[i][0] << 16, fields[i][1], fields[i][2]};
        std::memcpy(descriptors[mode == "dispatch-swapped" ? 1 - i : i], view, sizeof(view));
    }
    void* title_readback = nullptr;
    void* title_placement = nullptr;
    void* title_upload = nullptr;
    void* uav_source = nullptr;
    void* uav_staging = nullptr;
    unsigned char* source = nullptr;
    if (title_input || view_tokens) {
        const D3D12_RESOURCE_DESC output_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, sizeof(input), 1, 1, 1,
            DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        title_readback = make_committed(device, D3D12_HEAP_TYPE_READBACK, output_desc, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!title_readback) return 71;
    }
    if (title_input) {
        if (large_input || ring_input || protected_input) {
            source = static_cast<unsigned char*>(VirtualAlloc(nullptr, input_bytes + 0x1000,
                MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
            if (source && ((large_input && !alias_input) || ring_input)) source += 16;  // deliberately cannot alias a placed heap
        } else {
            source = static_cast<unsigned char*>(HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, input_bytes));
        }
        if (!source) return 70;
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, input_bytes + (gpu_input ? gpu_offset : 0), 1, 1, 1,
            DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        if ((large_input || ring_input) &&
            FAILED(static_cast<HRESULT>(invoke(device, 133, word(source), word(&desc), D3D12_RESOURCE_STATE_COMMON, 0,
                word(&__uuidof(ID3D12Resource)), word(&title_placement))))) return 71;

        void* records = source;
        if (gpu_input) {
            title_upload = make_committed(device, D3D12_HEAP_TYPE_UPLOAD, desc, D3D12_RESOURCE_STATE_GENERIC_READ);
            if (!title_upload || FAILED(static_cast<HRESULT>(invoke(title_upload, 8, 0, 0, word(&records)))) || !records)
                return 71;
        }
        for (UINT record = 0; record < 3; ++record) {
            UINT values[4];
            for (UINT i = 0; i < 4; ++i) values[i] = (record + 1) * 37 + i * 9;
            std::memcpy(static_cast<unsigned char*>(records) + record_offsets[record], values, sizeof(values));
        }
        if (gpu_input) invoke(title_upload, 9, 0, 0);
        if (protected_input) {
            DWORD old_protection = 0;
            if (!VirtualProtect(source + 0x1000, 0x1000, PAGE_NOACCESS, &old_protection)) return 71;
        }

        const UINT64 address = reinterpret_cast<UINT64>(source) + (gpu_input ? gpu_offset : 0);
        descriptors[0][0] = static_cast<UINT>(address);
        descriptors[0][1] = static_cast<UINT>(address >> 32) | 4u << 16;
        descriptors[0][2] = input_bytes / 4;
    }
    const UINT64 between = reinterpret_cast<UINT64>(memory) + 0x8000;
    // dispatch-input-snapshot: the input words are written into the memory the third buffer is placed over, so they
    // exist only in title memory (the desktop copy of a buffer that does not alias holds zeros); dispatch-input-copied
    // leaves that memory zero.
    if (mode == "dispatch-input-snapshot") std::memcpy(reinterpret_cast<void*>(between), input, sizeof(input));
    if (mode == "dispatch-input-copied" || mode == "dispatch-input-snapshot") {
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, words * sizeof(UINT), 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        void* copied = nullptr;
        if (FAILED(static_cast<HRESULT>(invoke(device, 133, word(reinterpret_cast<void*>(between)), word(&desc), D3D12_RESOURCE_STATE_COMMON, 0,
                word(&__uuidof(ID3D12Resource)), word(&copied))))) return 71;
    }
    if (mode == "dispatch-input-empty") {
        const UINT view[4] = {0x32c, 0, 0, 0};
        std::memcpy(descriptors[0], view, sizeof(view));
    } else if (mode == "dispatch-input-null") {
        std::memset(descriptors[0], 0, sizeof(descriptors[0]));  // an unbound buffer: all four words zero
    } else if (mode == "dispatch-input-past") {
        descriptors[0][2] = words + 4;
    } else if (mode == "dispatch-input-unplaced" || mode == "dispatch-input-copied" || mode == "dispatch-input-snapshot") {
        descriptors[0][0] = static_cast<UINT>(between);
        descriptors[0][1] = static_cast<UINT>(between >> 32) | 4u << 16;
    }
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
    void *heap = nullptr, *queue = nullptr, *allocator = nullptr, *list = nullptr, *fence = nullptr;
    // dispatch-sampled-compute records the same calls on a compute list and queue.
    const D3D12_COMMAND_LIST_TYPE list_type = mode == "dispatch-sampled-compute" ? D3D12_COMMAND_LIST_TYPE_COMPUTE : D3D12_COMMAND_LIST_TYPE_DIRECT;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{list_type};
    const UINT list_desc[]{static_cast<UINT>(list_type), 2, 1, 0};
    if (FAILED(static_cast<HRESULT>(invoke(device, 14, word(&heap_desc), word(&__uuidof(ID3D12DescriptorHeap)), word(&heap)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 8, word(&queue_desc), word(&__uuidof(ID3D12CommandQueue)), word(&queue)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 9, list_type, word(&__uuidof(ID3D12CommandAllocator)), word(&allocator)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 73, word(list_desc), word(allocator), 0, word(&__uuidof(ID3D12GraphicsCommandList)), word(&list)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 36, 0, 0, word(&__uuidof(ID3D12Fence)), word(&fence))))) return 72;
    const StreamWord cpu = invoke(heap, 9), gpu = invoke(heap, 10);
    if (!cpu || !gpu) return 73;
    if (mode == "dispatch-indirect-output-alias") {
        const UINT64 address = reinterpret_cast<UINT64>(memory) + 0x20000;
        const UINT alias[4] = {static_cast<UINT>(address), static_cast<UINT>(address >> 32) | (4u << 16), 5, 0};
        std::memcpy(descriptors[1], alias, sizeof(alias));
    }
    std::memcpy(reinterpret_cast<void*>(cpu), descriptors, sizeof(descriptors));
    const bool heap_record = mode.rfind("dispatch-heap", 0) == 0;
    if (heap_record)
        for (const UINT at : {0u, 32u, 64u})  // dispatch-heap-word-null reads record 0, so record 0 holds the input
            std::memcpy(reinterpret_cast<void*>(cpu + at), descriptors[at != 64 && !(at == 0 && mode == "dispatch-heap-word-null")],
                        sizeof(descriptors[0]));
    const bool formatted_indexed = mode == "dispatch-heap-format-indexed";
    const bool formatted_heap = mode.rfind("dispatch-heap-format-", 0) == 0;
    if (formatted_heap && mode != "dispatch-heap-format-plain") {
        DXGI_FORMAT format;
        UINT element;
        UINT mapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (formatted_indexed || mode == "dispatch-heap-format-half") {
            format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            element = 8;
        } else if (mode == "dispatch-heap-format-packed") {
            format = DXGI_FORMAT_R10G10B10A2_UNORM;
            element = 4;
        } else if (mode == "dispatch-heap-format-uint") {
            format = DXGI_FORMAT_R8G8B8A8_UINT;
            element = 4;
            mapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(2, 0, 3, 1);
        } else if (mode == "dispatch-heap-format-unsupported") {
            format = DXGI_FORMAT_R16G16B16A16_UINT;
            element = 8;
        } else {
            return 2;
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC view{format, D3D12_SRV_DIMENSION_BUFFER, mapping};
        view.Buffer = {0, 32u / element, 0, D3D12_BUFFER_SRV_FLAG_NONE};
        invoke(device, 18, word(placed[0]), word(&view), cpu + 64);
    }
    const bool heap_word = mode.rfind("dispatch-heap-word", 0) == 0;
    const bool dynamic_heap = mode.rfind("dispatch-heap-dynamic", 0) == 0;
    const UINT constants[3] = {title_input ? record_offsets[1] : 0x01234567u,
        title_input ? record_offsets[2] : 0x9e3779b9u, mode == "dispatch-heap-outside" ? 0x07ffffffu : heap_word ? 1u : 2u};
    const UINT graphics[2] = {0xdeadbeef, 0xfeedface};
    const auto bind = [&](void* list) {
        invoke(list, 32, 1, word(&heap));
        invoke(list, 25, word(slot67_pipeline));
        invoke(list, 33, word(mode == "dispatch-wrong-root" ? slot67_other_root : slot67_root));
    };
    if (FAILED(static_cast<HRESULT>(invoke(list, 139, word(allocator))))) return 74;
    Slot67Indirect record;
    if (indirect) record.prepare(adapter, device, list, memory + 0x20000 / sizeof(UINT), groups);
    bind(list);
    if (mode.rfind("dispatch-texel-", 0) == 0) {
        const auto format = mode == "dispatch-texel-float" ? DXGI_FORMAT_R32_FLOAT :
                            mode == "dispatch-texel-sint" ? DXGI_FORMAT_R32_SINT : DXGI_FORMAT_R32_UINT;
        if (!bind_identifier_fixture(device, list, cpu + 64, format, identifier_fixture_dimension())) return 80;
    }
    if (mode.rfind("dispatch-sampled", 0) == 0 || mode == "dispatch-input-token" || view_tokens) {
        const UINT width = view_tokens || mode == "dispatch-sampled-2x1" ? 2 : 1;
        const UINT16 mips = view_tokens ? 2 : 1;
        const D3D12_RESOURCE_DESC texture_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, 1, 1, mips, DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}};
        const D3D12_RESOURCE_DESC upload_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, view_tokens ? 768u : 256u, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        void* const texture = make_committed(device, D3D12_HEAP_TYPE_DEFAULT, texture_desc, D3D12_RESOURCE_STATE_COMMON);
        void* const upload = make_committed(device, D3D12_HEAP_TYPE_UPLOAD, upload_desc, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* texel = nullptr;
        if (!texture || !upload || FAILED(static_cast<HRESULT>(invoke(upload, 8, 0, 0, word(&texel)))) || !texel) return 80;
        // The program reads red: 1.0f. The 2x1 texture's left texel has red 0, so only a linear filter that wraps
        // reads 0.5 at the left edge (point or clamp read 0).
        unsigned char rgba[8] = {255, 64, 128, 32, 255, 64, 128, 32};
        if (width == 2) rgba[0] = 0;
        std::memcpy(texel, rgba, sizeof(rgba));
        if (view_tokens) {
            const unsigned char mip[] = {17, 93, 201, 45};
            std::memcpy(static_cast<unsigned char*>(texel) + 512, mip, sizeof(mip));
        }
        invoke(upload, 9, 0, 0);
        D3D12_TEXTURE_COPY_LOCATION image{static_cast<ID3D12Resource*>(texture), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION source{static_cast<ID3D12Resource*>(upload), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        source.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, width, 1, 1, 256}};
        invoke(list, 16, word(&image), 0, 0, 0, word(&source), 0);
        D3D12_SHADER_RESOURCE_VIEW_DESC view{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
        view.Texture2D.MipLevels = 1;
        if (view_tokens) {
            image.SubresourceIndex = 1;
            source.PlacedFootprint = {512, {DXGI_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, 256}};
            invoke(list, 16, word(&image), 0, 0, 0, word(&source), 0);
            view.Texture2D.MostDetailedMip = 1;
        }
        if (mode != "dispatch-sampled-unbound") invoke(device, 18, word(texture), word(&view), cpu + (mode == "dispatch-input-token" ? 0 : 64));
    }
    if (view_tokens) {
        if (mode == "dispatch-input-view-tokens-wrong-kind") {
            std::memcpy(reinterpret_cast<void*>(cpu), reinterpret_cast<const void*>(cpu + 64), 32);
        } else {
            D3D12_UNORDERED_ACCESS_VIEW_DESC view{DXGI_FORMAT_R32_TYPELESS, D3D12_UAV_DIMENSION_BUFFER};
            view.Buffer.FirstElement = mode == "dispatch-input-view-tokens-shifted" ? 12 : 8;
            view.Buffer.NumElements = 4;
            view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            if (mode == "dispatch-input-view-tokens-structured") {
                // Eight-byte records: the program reads the stride and record count the descriptor would have held.
                view.Format = DXGI_FORMAT_UNKNOWN;
                view.Buffer.FirstElement = 4;
                view.Buffer.StructureByteStride = 8;
                view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
            }
            invoke(device, 19, word(placed[1]), 0, word(&view), cpu);
        }
        if (mode == "dispatch-input-view-tokens-uav-input") {
            // A GPU-filled buffer the title bound through an unordered access view where the program reads a shader resource.
            const D3D12_RESOURCE_DESC source_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, sizeof(input), 1, 1, 1, DXGI_FORMAT_UNKNOWN,
                {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
            const D3D12_RESOURCE_DESC staging_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, sizeof(input), 1, 1, 1, DXGI_FORMAT_UNKNOWN,
                {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
            uav_source = make_committed(device, D3D12_HEAP_TYPE_DEFAULT, source_desc, D3D12_RESOURCE_STATE_COMMON);
            uav_staging = make_committed(device, D3D12_HEAP_TYPE_UPLOAD, staging_desc, D3D12_RESOURCE_STATE_GENERIC_READ);
            void* staged = nullptr;
            if (!uav_source || !uav_staging || FAILED(static_cast<HRESULT>(invoke(uav_staging, 8, 0, 0, word(&staged)))) || !staged) return 71;
            for (UINT i = 0; i < words; ++i) static_cast<UINT*>(staged)[i] = 0xa0000000u + i * 0x101u;
            invoke(uav_staging, 9, 0, 0);
            D3D12_UNORDERED_ACCESS_VIEW_DESC read{DXGI_FORMAT_R32_TYPELESS, D3D12_UAV_DIMENSION_BUFFER};
            read.Buffer.FirstElement = 4;  // 16 bytes in
            read.Buffer.NumElements = 8;
            read.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            invoke(device, 19, word(uav_source), 0, word(&read), cpu + 64);
        }
    }
    if (mode == "dispatch-input-null-token") {
        D3D12_SHADER_RESOURCE_VIEW_DESC view{DXGI_FORMAT_R32_TYPELESS, D3D12_SRV_DIMENSION_BUFFER,
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
        view.Buffer = {0, words, 0, D3D12_BUFFER_SRV_FLAG_RAW};
        invoke(device, 18, 0, word(&view), cpu);
        // The 32-byte token overlaps the following 16-byte table descriptor.
        std::memcpy(reinterpret_cast<void*>(cpu + sizeof(descriptors[0])), descriptors[1], sizeof(descriptors[1]));
    }
    if (mode == "dispatch-input-memory-view") {
        // The input is a buffer view (device slot 70) of memory no buffer is placed over, 16 bytes in; its token
        // replaces the input descriptor and the words are written after the view exists.
        auto* const unplaced = static_cast<UINT*>(VirtualAlloc(nullptr, 0x10000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!unplaced) return 70;
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 0x10000, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        struct {
            D3D12_SHADER_RESOURCE_VIEW_DESC view;
            UINT64 address;
        } input{{DXGI_FORMAT_UNKNOWN, D3D12_SRV_DIMENSION_BUFFER, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING},
                reinterpret_cast<UINT64>(unplaced)};
        input.view.Buffer = {4, words, sizeof(UINT), D3D12_BUFFER_SRV_FLAG_NONE};
        invoke(device, 70, word(&desc), word(&input), cpu);
        // The token is wider than a table descriptor: the output descriptor after it is written again.
        std::memcpy(reinterpret_cast<void*>(cpu + sizeof(descriptors[0])), descriptors[1], sizeof(descriptors[1]));
        std::memcpy(unplaced + 4, buffers[0], words * sizeof(UINT));
    }
    if (mode != "dispatch-missing-word") invoke(list, 37, 0, constants[0], 0);
    invoke(list, 40, 0, 2, word(graphics), 0);
    invoke(list, 39, 0, 1, word(&constants[1]), 1);
    if (mode == "dispatch-root-changed") invoke(list, 33, word(slot67_other_root));
    if (mode == "dispatch-root-changed" || mode == "dispatch-same-root") invoke(list, 33, word(slot67_root));
    if (mode == "dispatch-reset") {
        if (FAILED(static_cast<HRESULT>(invoke(list, 9))) || FAILED(static_cast<HRESULT>(invoke(list, 139, word(allocator))))) return 74;
        bind(list);
    }
    if (heap_record) invoke(list, 39, 0, 1, word(&constants[2]), 2);
    if (dynamic_heap) {
        auto* const index = reinterpret_cast<UINT*>(invoke(list, 148, 16, 256));
        if (!index) return 81;
        index[0] = 3;
        index[1] = mode.find("-outside") != std::string::npos ? 4 : 2;
        invoke(list, mode.find("-cbv") != std::string::npos ? 41 : 43, 2, word(index));
        const UINT64 index_address = reinterpret_cast<UINT64>(index);
        const UINT index_view[4]{static_cast<UINT>(index_address), static_cast<UINT>(index_address >> 32), 16, 0};
        std::memcpy(reinterpret_cast<void*>(cpu), index_view, sizeof(index_view));
        for (UINT slot : {2u, 3u}) {
            auto* const data = reinterpret_cast<UINT*>(invoke(list, 148, 16, 256));
            if (!data) return 81;
            *data = slot == 2 ? 0x428a6000u : 0x42aa6000u;
            const UINT64 address = reinterpret_cast<UINT64>(data);
            const UINT raw[4]{static_cast<UINT>(address), static_cast<UINT>(address >> 32), 16, 0};
            std::memcpy(reinterpret_cast<void*>(cpu + slot * 32), raw, sizeof(raw));
        }
    }
    if (mode == "dispatch-heap-word-null")
        std::memset(reinterpret_cast<void*>(cpu + 96), 0, 16);  // the index buffer's descriptor stays null: the word reads as 0
    else if (heap_word || formatted_indexed) {
        // The index buffer at table byte 96 lives in the list's upload memory. Only the expected
        // byte offset holds record 2. The other words name an output descriptor.
        constexpr UINT bytes = 0x400;
        const UINT at = formatted_indexed ? 52 : 0x270;
        auto* const index = static_cast<UINT*>(reinterpret_cast<void*>(invoke(list, 148, bytes, 256)));
        if (!index) return 81;
        std::memset(index, 0, bytes);
        index[0x120 / 4] = index[0x150 / 4] = 1;
        index[at / 4] = mode == "dispatch-heap-word-outside" ? 0x07ffffffu : 2u;
        const UINT64 address = reinterpret_cast<UINT64>(mode == "dispatch-heap-word-placed" ? buffers[0] : index);
        const UINT view[4] = {static_cast<UINT>(address), static_cast<UINT>(address >> 32),
                              mode == "dispatch-heap-word-past" ? at : mode == "dispatch-heap-word-placed" ? words * 4 : bytes, 0};
        std::memcpy(reinterpret_cast<void*>(cpu + 96), view, sizeof(view));
    }
    invoke(list, 35, 1, gpu);
    if (dynamic_heap) invoke(list, 14, 1, 1, 1);
    else if (indirect) record.execute(list, mode == "dispatch-indirect-multi" ? 2 : 1,
                                 mode == "dispatch-indirect-counted", mode == "dispatch-indirect-outside");
    else if (mode == "dispatch-reference-2" || mode == "dispatch-reference-5")
        invoke(list, 14, mode == "dispatch-reference-2" ? 2 : 5, 1, 1);
    else if (mode == "dispatch-nolock-burst") {
        const auto before = compute_refusal_stats();
        for (UINT i = 0; i < 10000; ++i) {
            if (i == 5000) Sleep(300);  // Cross the legacy retry interval.
            invoke(list, 14, 3, 2, 2);
        }
        const auto after = compute_refusal_stats();
        std::printf("compute refusal window: manifest lookups: repeated reads %lld, draw_lock holds +%lld, untranslated +%ld\n",
                    after.repeated_reads, after.holds - before.holds, after.untranslated - before.untranslated);
    } else if (mode == "dispatch-nolock-retry") {
        void* second_allocator = nullptr;
        void* second_list = nullptr;
        if (FAILED(static_cast<HRESULT>(invoke(device, 9, list_type, word(&__uuidof(ID3D12CommandAllocator)), word(&second_allocator)))) ||
            FAILED(static_cast<HRESULT>(invoke(device, 73, word(list_desc), word(second_allocator), 0,
                                              word(&__uuidof(ID3D12GraphicsCommandList)), word(&second_list)))) ||
            FAILED(static_cast<HRESULT>(invoke(second_list, 139, word(second_allocator))))) return 82;
        bind(second_list);
        invoke(second_list, 37, 0, constants[0], 0);
        invoke(second_list, 39, 0, 1, word(&constants[1]), 1);
        invoke(second_list, 35, 1, gpu);
        std::atomic<bool> ready{false}, start{false};
        std::thread other([&] {
            ready.store(true, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) SwitchToThread();
            invoke(second_list, 14, 3, 2, 2);
        });
        while (!ready.load(std::memory_order_acquire)) SwitchToThread();
        start.store(true, std::memory_order_release);
        invoke(list, 14, 3, 2, 2);
        other.join();
        if (FAILED(static_cast<HRESULT>(invoke(second_list, 9)))) return 83;
        invoke(queue, 10, 1, word(&second_list));
    }
    else if (mode == "dispatch-bench") {
        // The speed bench: X2W_TEST_BENCH_CALLS (default 3000) rounds of the cliff's pattern per recording thread, each a descriptor
        // table (slot 35), a root constant (slot 39) and a Dispatch (slot 14), every call timed here. X2W_TEST_BENCH_THREADS (default 1)
        // records that many lists at once, one per thread, so the share of waiting for the stand-in's global lock shows. Prints the totals
        // in the shape of the stand-in's own `command-list slots over 20 ms` line, so tools/startup/slot_compare.py can diff a bench run
        // against a tester log.
        char text[16]{};
        GetEnvironmentVariableA("X2W_TEST_BENCH_CALLS", text, sizeof(text));
        const int rounds = text[0] ? std::atoi(text) : 3000;
        char thread_text[16]{};
        GetEnvironmentVariableA("X2W_TEST_BENCH_THREADS", thread_text, sizeof(thread_text));
        const int threads = thread_text[0] ? std::atoi(thread_text) : 1;
        if (threads < 1 || threads > 16) return 2;
        std::vector<void*> lists{list};
        for (int t = 1; t < threads; ++t) {
            void *more_allocator = nullptr, *more_list = nullptr;
            if (FAILED(static_cast<HRESULT>(invoke(device, 9, list_type, word(&__uuidof(ID3D12CommandAllocator)), word(&more_allocator)))) ||
                FAILED(static_cast<HRESULT>(invoke(device, 73, word(list_desc), word(more_allocator), 0,
                                                   word(&__uuidof(ID3D12GraphicsCommandList)), word(&more_list))))) return 72;
            if (FAILED(static_cast<HRESULT>(invoke(more_list, 139, word(more_allocator))))) return 74;
            invoke(more_list, 32, 1, word(&heap));
            invoke(more_list, 25, word(slot67_pipeline));
            invoke(more_list, 33, word(slot67_root));
            invoke(more_list, 37, 0, constants[0], 0);
            invoke(more_list, 40, 0, 2, word(graphics), 0);
            invoke(more_list, 39, 0, 1, word(&constants[1]), 1);
            lists.push_back(more_list);
        }
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        std::vector<std::array<double, 3>> per_thread(threads);
        const auto work = [&](int t) {
            LARGE_INTEGER a, b;
            double* ms = per_thread[t].data();
            const auto timed = [&](int which, auto&& call) {
                QueryPerformanceCounter(&a);
                call();
                QueryPerformanceCounter(&b);
                ms[which] += 1000.0 * static_cast<double>(b.QuadPart - a.QuadPart) / static_cast<double>(frequency.QuadPart);
            };
            void* const mine = lists[t];
            for (int i = 0; i < rounds; ++i) {
                timed(0, [&] { invoke(mine, 35, 1, gpu); });
                timed(1, [&] { invoke(mine, 39, 0, 1, word(&constants[1]), 1); });
                timed(2, [&] { invoke(mine, 14, 3, 2, 2); });
            }
        };
        LARGE_INTEGER wall_start, wall_end;
        QueryPerformanceCounter(&wall_start);
        std::vector<std::thread> workers;
        for (int t = 1; t < threads; ++t) workers.emplace_back(work, t);
        work(0);
        for (auto& w : workers) w.join();
        QueryPerformanceCounter(&wall_end);
        double ms[3] = {};
        for (const auto& p : per_thread)
            for (int k = 0; k < 3; ++k) ms[k] += p[k];
        const double wall_ms = 1000.0 * static_cast<double>(wall_end.QuadPart - wall_start.QuadPart) / static_cast<double>(frequency.QuadPart);
        const long long calls = static_cast<long long>(rounds) * threads;
        std::printf("x2w-bench threads %d, rounds %d each, wall %.0f ms: slot35 %.2f us/call, slot39 %.2f us/call, slot14 %.2f us/call\n", threads,
                    rounds, wall_ms, ms[0] * 1000.0 / calls, ms[1] * 1000.0 / calls, ms[2] * 1000.0 / calls);
        std::fprintf(stderr, "x2w-d3d12 command-list slots over 20 ms: slot35 %.0f ms/%lld slot39 %.0f ms/%lld slot14 %.0f ms/%lld\n", ms[0],
                     calls, ms[1], calls, ms[2], calls);
        for (std::size_t t = 1; t < lists.size(); ++t) {
            invoke(lists[t], 9);
            invoke(queue, 10, 1, word(&lists[t]));
        }
    } else if (formatted_heap) invoke(list, 14, 1, 1, 1);
    else if (view_tokens) {
        if (uav_source) invoke(list, 15, word(uav_source), 0, word(uav_staging), 0, sizeof(input));
        invoke(list, 14, 1, 1, 1);
        invoke(list, 15, word(title_readback), 0, word(placed[1]), 0, sizeof(input));
    } else if (title_input) {
        if (ring_input) {
            const bool fill_overflow = mode == "dispatch-input-title-ring-full";
            if (x2w_d3d12_fixture_fill_upload_ring(list, 16u << 10, fill_overflow) != (fill_overflow ? 5u : 4u))
                return 81;
        }
        if (gpu_input) invoke(list, 15, word(title_placement), gpu_offset, word(title_upload), 0, input_bytes);
        invoke(list, 14, 1, 1, 1);
        if (!protected_input && !alias_input) std::memset(source, 0, 16);  // snapshot must have happened at record time
        invoke(list, 15, word(title_readback), 0, word(placed[1]), 0, sizeof(input));
    } else if (mode != "dispatch-none") invoke(list, 14, 3, 2, 2);
    if (mode == "dispatch-input-empty") invoke(list, 14, 3, 2, 2);
    if (FAILED(static_cast<HRESULT>(invoke(list, 9)))) return 75;
    invoke(queue, 10, 1, word(&list));
    if (FAILED(static_cast<HRESULT>(invoke(queue, 14, word(fence), 1)))) return 76;
    for (const ULONGLONG deadline = GetTickCount64() + 10000; invoke(fence, 8) < 1;) {
        if (GetTickCount64() >= deadline) return 78;
        Sleep(1);
    }
    if (indirect && !record.check_record(groups)) return 80;
    for (UINT i = 0; i < words; ++i)
        if (buffers[0][i] != input[i]) return 79;  // the input must stay as written
    x2w_d3d12_compute_census();
    char path[MAX_PATH];
    const DWORD n = GetEnvironmentVariableA("X2W_TEST_DISPATCH_OUT", path, MAX_PATH);
    if (!n || n >= MAX_PATH) return 2;
    void* result = buffers[1];
    if (title_readback && (FAILED(static_cast<HRESULT>(invoke(title_readback, 8, 0, 0, word(&result)))) || !result)) return 2;
    std::ofstream(path, std::ios::binary).write(static_cast<const char*>(result), sizeof(input));
    if (title_readback) invoke(title_readback, 9, 0, 0);
    return 0;
}

// The image store program of startup_compute_image_store_d16_check.py through the stand-in's own list: a committed
// 32x8 RGBA16_FLOAT texture (device slot 127) filled with sentinel halves by CopyTextureRegion (list slot 16), an
// unordered access view of it (device slot 19) whose token the stand-in writes into the title heap at table byte 64,
// three root constants, Dispatch of 2x1x1 groups, and the texture copied to a readback buffer whose 1024 halves go
// to X2W_TEST_DISPATCH_OUT. texture-store-none records no Dispatch; texture-store-denied creates the texture
// without unordered access, so the stand-in records its view as not created. texture-store-uint makes the texture
// and its view RGBA16_UINT; texture-store-occupied then asks for a view of a texture without unordered access at the same
// handle, which keeps the first view; texture-store-self also creates a shader resource view of the texture at table byte 0.
static int texture_slot67(void* device, const std::string& mode)
{
    constexpr UINT width = 32, height = 8;
    const bool fp32 = mode == "texture-array" || mode == "texture-array-2d";
    const UINT layers = mode == "texture-array" || mode == "texture-array-d16" ? 6 : 1;
    const UINT pitch = width * (fp32 ? 16 : 8), slice_bytes = pitch * height, bytes = slice_bytes * layers;
    const auto word = [](const void* value) { return reinterpret_cast<StreamWord>(value); };
    const auto make = [&](D3D12_HEAP_TYPE type, const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES state) {
        return make_committed(device, type, desc, state);
    };
    const DXGI_FORMAT format = fp32 ? DXGI_FORMAT_R32G32B32A32_FLOAT :
        mode == "texture-store-uint" ? DXGI_FORMAT_R16G16B16A16_UINT : DXGI_FORMAT_R16G16B16A16_FLOAT;
    const D3D12_RESOURCE_DESC texture_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, static_cast<UINT16>(layers), 1, format,
        {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN,
        mode == "texture-store-denied" ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    const D3D12_RESOURCE_DESC buffer_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
    void* const texture = make(D3D12_HEAP_TYPE_DEFAULT, texture_desc, D3D12_RESOURCE_STATE_COMMON);
    void* const upload = make(D3D12_HEAP_TYPE_UPLOAD, buffer_desc, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* const readback = make(D3D12_HEAP_TYPE_READBACK, buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST);
    void* memory = nullptr;
    if (!texture || !upload || !readback || FAILED(static_cast<HRESULT>(invoke(upload, 8, 0, 0, word(&memory)))) || !memory) return 71;
    if (fp32) {
        for (UINT layer = 0; layer < layers; ++layer)
            for (UINT i = 0; i < slice_bytes / 4; ++i)
                static_cast<UINT*>(memory)[layer * slice_bytes / 4 + i] = 0xbf000000u | (layer << 12) | i;
    } else {
        for (UINT i = 0; i < bytes / 2; ++i) static_cast<UINT16*>(memory)[i] = static_cast<UINT16>(0x5000 + i);
    }
    invoke(upload, 9, 0, 0);
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
    void *heap = nullptr, *queue = nullptr, *allocator = nullptr, *list = nullptr, *fence = nullptr;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    if (FAILED(static_cast<HRESULT>(invoke(device, 14, word(&heap_desc), word(&__uuidof(ID3D12DescriptorHeap)), word(&heap)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 8, word(&queue_desc), word(&__uuidof(ID3D12CommandQueue)), word(&queue)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, word(&__uuidof(ID3D12CommandAllocator)), word(&allocator)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 73, word(list_desc), word(allocator), 0, word(&__uuidof(ID3D12GraphicsCommandList)), word(&list)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 36, 0, 0, word(&__uuidof(ID3D12Fence)), word(&fence))))) return 72;
    const StreamWord cpu = invoke(heap, 9), gpu = invoke(heap, 10);
    if (!cpu || !gpu) return 73;
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{format, D3D12_UAV_DIMENSION_TEXTURE2D};
    if (layers > 1) {
        view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        view.Texture2DArray = {0, 1, 4, 0}; // four selected slices, with untouched neighbors on both sides
    }
    if (mode != "texture-store-copied") invoke(device, 19, word(texture), 0, word(&view), cpu + 64);
    if (mode == "texture-store-copied") {
        // The view is created at another handle and its token copied into this slot by memcpy, as a title does; a
        // refused creation at this slot then keeps that copied token.
        invoke(device, 19, word(texture), 0, word(&view), cpu);
        std::memcpy(reinterpret_cast<void*>(cpu + 64), reinterpret_cast<void*>(cpu), 32);
        D3D12_RESOURCE_DESC plain = texture_desc;
        plain.Flags = D3D12_RESOURCE_FLAG_NONE;
        void* const other = make(D3D12_HEAP_TYPE_DEFAULT, plain, D3D12_RESOURCE_STATE_COMMON);
        if (!other) return 79;
        invoke(device, 19, word(other), 0, word(&view), cpu + 64);
    } else if (mode == "texture-store-occupied") {
        // A second creation at the same handle, over a texture without unordered access, is refused and leaves the first view.
        D3D12_RESOURCE_DESC plain = texture_desc;
        plain.Flags = D3D12_RESOURCE_FLAG_NONE;
        void* const other = make(D3D12_HEAP_TYPE_DEFAULT, plain, D3D12_RESOURCE_STATE_COMMON);
        if (!other) return 79;
        invoke(device, 19, word(other), 0, word(&view), cpu + 64);
    }
    D3D12_SHADER_RESOURCE_VIEW_DESC read{format, D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
    read.Texture2D.MipLevels = 1;
    if (mode == "texture-store-self") invoke(device, 18, word(texture), word(&read), cpu);
    D3D12_TEXTURE_COPY_LOCATION image{static_cast<ID3D12Resource*>(texture), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    D3D12_TEXTURE_COPY_LOCATION source{static_cast<ID3D12Resource*>(upload), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    source.PlacedFootprint = {0, {format, width, height, 1, pitch}};
    D3D12_TEXTURE_COPY_LOCATION result = source;
    result.pResource = static_cast<ID3D12Resource*>(readback);
    const UINT constants[3] = {1, 2, 3};
    if (FAILED(static_cast<HRESULT>(invoke(list, 139, word(allocator))))) return 74;
    invoke(list, 32, 1, word(&heap));
    for (UINT layer = 0; layer < layers; ++layer) {
        image.SubresourceIndex = layer;
        source.PlacedFootprint.Offset = UINT64{layer} * slice_bytes;
        invoke(list, 16, word(&image), 0, 0, 0, word(&source), 0);
    }
    invoke(list, 25, word(slot67_pipeline));
    invoke(list, 33, word(slot67_root));
    invoke(list, 39, 0, 3, word(constants), 0);
    invoke(list, 35, 1, gpu);
    if (mode != "texture-store-none") invoke(list, 14, 2, 1, 1);
    for (UINT layer = 0; layer < layers; ++layer) {
        image.SubresourceIndex = layer;
        result.PlacedFootprint.Offset = UINT64{layer} * slice_bytes;
        invoke(list, 16, word(&result), 0, 0, 0, word(&image), 0);
    }
    if (FAILED(static_cast<HRESULT>(invoke(list, 9)))) return 75;
    invoke(queue, 10, 1, word(&list));
    if (FAILED(static_cast<HRESULT>(invoke(queue, 14, word(fence), 1)))) return 76;
    for (const ULONGLONG deadline = GetTickCount64() + 10000; invoke(fence, 8) < 1;) {
        if (GetTickCount64() >= deadline) return 78;
        Sleep(1);
    }
    x2w_d3d12_compute_census();
    char path[MAX_PATH];
    const DWORD n = GetEnvironmentVariableA("X2W_TEST_DISPATCH_OUT", path, MAX_PATH);
    memory = nullptr;
    if (!n || n >= MAX_PATH || FAILED(static_cast<HRESULT>(invoke(readback, 8, 0, 0, word(&memory)))) || !memory) return 2;
    std::ofstream(path, std::ios::binary).write(static_cast<const char*>(memory), bytes);
    invoke(readback, 9, 0, 0);
    return 0;
}

static bool create_slot67(void* device, const char* code_path, const char* data_path, const char* state_path, const char* mode)
{
    std::ifstream code_file(code_path, std::ios::binary), data_file(data_path, std::ios::binary), state_file(state_path, std::ios::binary);
    std::vector<char> code{std::istreambuf_iterator<char>(code_file), {}};
    std::vector<char> data{std::istreambuf_iterator<char>(data_file), {}};
    std::vector<char> state{std::istreambuf_iterator<char>(state_file), {}};
    if (code.empty() || data.empty() || state.empty() || !stream_root_signature()) return false;
    void* root = stream_root_signature();
    slot67_other_root = root;
    char root_path[MAX_PATH];
    const DWORD n = GetEnvironmentVariableA("X2W_TEST_COMPUTE_ROOT", root_path, MAX_PATH);
    if (n && n < MAX_PATH) {
        std::ifstream file(root_path, std::ios::binary);
        const std::vector<char> blob{std::istreambuf_iterator<char>(file), {}};
        using CreateRoot = HRESULT (WINAPI*)(void*, UINT, const void*, SIZE_T, REFIID, void**);
        if (blob.empty() || FAILED(reinterpret_cast<CreateRoot>((*static_cast<void***>(device))[16])(
                device, 0, blob.data(), blob.size(), __uuidof(ID3D12RootSignature), &root))) return false;
    }
    // Two pages: the first readable, the second PAGE_NOACCESS.
    auto* pages = static_cast<unsigned char*>(VirtualAlloc(nullptr, 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    DWORD old = 0;
    if (!pages || !VirtualProtect(pages + 0x1000, 0x1000, PAGE_NOACCESS, &old)) return false;
    unsigned char* guard = pages + 0x1000;
    const std::string m(mode);
    if (m == "code-limit" || m == "code-over") code.resize((1u << 20) + (m == "code-over"));
    if (m == "state-limit" || m == "state-over") state.resize(0x1000 + (m == "state-over"));
    std::vector<char> state_b{char(0xcd), char(0xab), 0x34, 0x12, 0, 0, char(0xff), char(0xff), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (m == "data-limit" || m == "data-over") data.resize((1u << 20) + (m == "data-over"));
    if (m == "stateb-limit" || m == "stateb-over") state_b.resize(0x1000 + (m == "stateb-over"));
    if (m == "data-pointer") {  // the first qword of the data blob points at state block A
        const UINT64 address = reinterpret_cast<UINT64>(state.data());
        if (data.size() < sizeof(address)) return false;
        std::memcpy(data.data(), &address, sizeof(address));
    }
    alignas(8) unsigned char desc[0x300]{};
    UINT64 words[12] = {reinterpret_cast<UINT64>(root), reinterpret_cast<UINT64>(state.data()), state.size(),
                        reinterpret_cast<UINT64>(state_b.data()), state_b.size(),
                        reinterpret_cast<UINT64>(code.data()), code.size(), reinterpret_cast<UINT64>(data.data()), data.size(),
                        reinterpret_cast<UINT64>(state.data()), reinterpret_cast<UINT64>(desc), 0xb};
    if (m == "code-zero") words[6] = 0;
    if (m == "data-zero") words[8] = 0;
    if (m == "no-root") words[0] = 0;
    if (m == "code-unreadable") words[5] = reinterpret_cast<UINT64>(guard);
    if (m == "state-unreadable") words[1] = reinterpret_cast<UINT64>(guard);
    if (m == "data-unreadable") words[7] = reinterpret_cast<UINT64>(guard);
    if (m == "stateb-unreadable") words[3] = reinterpret_cast<UINT64>(guard);
    if (m == "code-past-end") {
        std::memcpy(guard - code.size(), code.data(), code.size());
        words[5] = reinterpret_cast<UINT64>(guard - code.size());
        words[6] = code.size() + 1;
    }
    std::memcpy(desc, words, sizeof(words));
    const void* where = desc;
    if (m == "desc-unreadable") where = guard;
    using Create = HRESULT (WINAPI*)(void*, const void*, REFIID, void**);
    auto create = [&](const void* d) {
        void* pipeline = nullptr;
        const bool created = SUCCEEDED(reinterpret_cast<Create>((*static_cast<void***>(device))[67])(device, d, __uuidof(ID3D12PipelineState), &pipeline)) &&
               pipeline;
        slot67_pipeline = pipeline;
        slot67_root = root;
        return created;
    };
    if (m == "order") {
        if (!create(desc)) return false;
        UINT64 refused[12];
        std::memcpy(refused, words, sizeof(refused));
        refused[0] = 0;
        if (!create(refused)) return false;
        std::vector<char> other(data);
        other[0] ^= 1;
        words[7] = reinterpret_cast<UINT64>(other.data());
        std::memcpy(desc, words, sizeof(words));
        return create(desc);
    }
    for (int i = 0; i < (m == "valid" ? 2 : 1); ++i)
        if (!create(where)) return false;
    return true;
}

// present-flat / present-split / present-onset (two-tone for 6 s, then flat for 6 s): a 64x64 RGBA8 render target cleared (ClearRenderTargetView, list slot 52) to one gray (128), or to
// black with its left half cleared to white, then presented 600 times through queue slot 32 with X2W_D3D12_RENDER=1. The stand-in
// reads the picture back and logs "x2w-render presented ... mean level M/255, spread S" and, every 300th present, an
// "x2w-gray interval" line of counters.
static int present_gray(void* device, const std::string& mode)
{
    constexpr UINT size = 64, presents = 600;
    const auto word = [](const void* value) { return reinterpret_cast<StreamWord>(value); };
    const bool split = mode == "present-split", onset = mode == "present-onset";
    const D3D12_RESOURCE_DESC texture_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, size, size, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0},
        D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    void* const texture = make_committed(device, D3D12_HEAP_TYPE_DEFAULT, texture_desc, D3D12_RESOURCE_STATE_COMMON);
    if (!texture) return 80;
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } rtv_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0};
    void *rtv_heap = nullptr, *queue = nullptr, *allocator = nullptr, *list = nullptr, *fence = nullptr;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    const UINT list_desc[]{static_cast<UINT>(D3D12_COMMAND_LIST_TYPE_DIRECT), 2, 1, 0};
    if (FAILED(static_cast<HRESULT>(invoke(device, 14, word(&rtv_desc), word(&__uuidof(ID3D12DescriptorHeap)), word(&rtv_heap)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 8, word(&queue_desc), word(&__uuidof(ID3D12CommandQueue)), word(&queue)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, word(&__uuidof(ID3D12CommandAllocator)), word(&allocator)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 73, word(list_desc), word(allocator), 0, word(&__uuidof(ID3D12GraphicsCommandList)), word(&list)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 36, 0, 0, word(&__uuidof(ID3D12Fence)), word(&fence))))) return 72;
    const StreamWord handle = invoke(rtv_heap, 9);
    if (!handle) return 73;
    const D3D12_RENDER_TARGET_VIEW_DESC view{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RTV_DIMENSION_TEXTURE2D, {}};
    invoke(device, 20, word(texture), word(&view), handle);
    const float gray[4] = {128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f, 1.0f};
    const float black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    const float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const D3D12_RECT left_half{0, 0, static_cast<LONG>(size / 2), static_cast<LONG>(size)};
    UINT64 fence_value = 0;
    const auto submit = [&] {
        if (FAILED(static_cast<HRESULT>(invoke(list, 9)))) return 75;
        invoke(queue, 10, 1, word(&list));
        if (FAILED(static_cast<HRESULT>(invoke(queue, 14, word(fence), ++fence_value)))) return 76;
        for (const ULONGLONG deadline = GetTickCount64() + 10000; invoke(fence, 8) < fence_value;) {
            if (GetTickCount64() >= deadline) return 78;
            Sleep(1);
        }
        return 0;
    };
    const auto fill = [&](bool two_tone, const float* color) {
        if (FAILED(static_cast<HRESULT>(invoke(list, 139, word(allocator), 0)))) return 74;
        invoke(list, 52, handle, word(two_tone ? black : color));
        if (two_tone) invoke(list, 52, handle, word(white), 1, word(&left_half));
        return submit();
    };
    // Plane parameters: +0x00 frame token, +0x08 resource count, +0x10 pointer to the resources.
    static std::uint64_t resources[1];
    static std::uint64_t plane[9];
    static std::uint64_t present[8];
    resources[0] = word(texture);
    plane[1] = 1;
    plane[2] = word(resources);
    const auto show = [&](UINT count, DWORD pause_ms) {
        for (UINT i = 0; i < count; ++i) {
            if (FAILED(static_cast<HRESULT>(invoke(queue, 32, 1, word(plane), word(present))))) return 81;
            if (pause_ms) Sleep(pause_ms);
        }
        return 0;
    };
    if (mode == "present-recent-fps") {
        if (const int rc = fill(false, black)) return rc;
        if (const int rc = show(30, 0)) return rc;
        const float final_color[4] = {37.0f / 255.0f, 109.0f / 255.0f, 223.0f / 255.0f, 1.0f};
        if (const int rc = fill(false, final_color)) return rc;
        for (UINT i = 0; i < 30; ++i) {
            Sleep(250);
            if (const int rc = show(1, 0)) return rc;
        }

        DWORD pid = 0;
        HWND window = nullptr;
        while ((window = FindWindowExW(nullptr, window, L"X2WOutputWindow", nullptr)) != nullptr) {
            GetWindowThreadProcessId(window, &pid);
            if (pid == GetCurrentProcessId()) break;
        }
        wchar_t caption[256];
        if (!window || !GetWindowTextW(window, caption, 256)) return 82;
        std::printf("caption: %ls\n", caption);

        constexpr UINT pitch = size * 4, bytes = size * pitch;
        const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1,
            DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        void* const readback = make_committed(device, D3D12_HEAP_TYPE_READBACK, buffer, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!readback || FAILED(static_cast<HRESULT>(invoke(list, 139, word(allocator), 0)))) return 83;
        const D3D12_TEXTURE_COPY_LOCATION source{static_cast<ID3D12Resource*>(texture), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION dest{static_cast<ID3D12Resource*>(readback), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dest.PlacedFootprint = {0, {texture_desc.Format, size, size, 1, pitch}};
        invoke(list, 16, word(&dest), 0, 0, 0, word(&source), 0);
        if (const int rc = submit()) return rc;

        char path[MAX_PATH];
        const DWORD n = GetEnvironmentVariableA("X2W_TEST_PRESENT_OUT", path, MAX_PATH);
        void* pixels = nullptr;
        if (!n || n >= MAX_PATH || FAILED(static_cast<HRESULT>(invoke(readback, 8, 0, 0, word(&pixels)))) || !pixels) return 84;
        std::ofstream output(path, std::ios::binary);
        output.write(static_cast<const char*>(pixels), bytes);
        output.close();
        invoke(readback, 9, 0, 0);
        return output ? 0 : 85;
    }
    if (onset) {
        // Two-tone for 6 s, then a flat gray for 6 s: the picture turns flat once, about 6 s into the run.
        if (const int rc = fill(true, gray)) return rc;
        if (const int rc = show(30, 200)) return rc;
        if (const int rc = fill(false, gray)) return rc;
        return show(30, 200);
    }
    if (const int rc = fill(split, gray)) return rc;
    return show(presents, 0);
}

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 5 && (argc < 6 || argc > 9) && argc != 10) return 2;
    if (argc == 9 && std::strcmp(argv[4], "slot67") != 0) return 2;
    const bool draw = argc != 4;
    StartupBc7Input bc7{};
    if (argc == 10) {
        bc7 = {argv[6], static_cast<UINT>(std::strtoul(argv[7], nullptr, 10)),
               static_cast<UINT>(std::strtoul(argv[8], nullptr, 10)),
               std::strcmp(argv[9], "tiled") == 0};
        bc7.small4k = std::strcmp(argv[9], "small4k") == 0;
        bc7.dropmips = std::strcmp(argv[9], "dropmips") == 0;
        if (bc7.small4k ? bc7.width != 32 || bc7.height != 32
                        : (bc7.width != 64 || bc7.height != 64) && (bc7.width != 81 || bc7.height != 69)) return 2;
        if (std::strncmp(argv[9], "packed:", 7) == 0) bc7.packed = argv[9] + 7;
        else if (bc7.small4k || bc7.dropmips) {}
        else if (!bc7.tiled && std::strcmp(argv[9], "linear") != 0) return 2;
    }
    ComPtr<IDXGIAdapter> adapter;
    void* device = nullptr;
    void* pso = nullptr;
    const bool hardware = (draw && std::strcmp(argv[4], "hardware") == 0) ||
                          GetEnvironmentVariableW(L"X2W_TEST_DISPATCH_HARDWARE", nullptr, 0) != 0;
    const bool textured = bc7.blocks != nullptr || GetEnvironmentVariableW(L"X2W_TEST_IDENTIFIER_SRV", nullptr, 0) != 0 || GetEnvironmentVariableW(L"X2W_TEST_D16_SRV", nullptr, 0) != 0 ||
                          GetEnvironmentVariableW(L"X2W_TEST_FORMATTED_BUFFER", nullptr, 0) != 0 ||
                          GetEnvironmentVariableW(L"X2W_TEST_FORMATTED_VIEW", nullptr, 0) != 0 ||
                          GetEnvironmentVariableW(L"X2W_TEST_JOINED_TABLE", nullptr, 0) != 0 ||
                          GetEnvironmentVariableW(L"X2W_TEST_COPIED_VERTEX_SRV_PAIR", nullptr, 0) != 0 ||
                          GetEnvironmentVariableW(L"X2W_TEST_PIXEL_UAV", nullptr, 0) != 0;
    if (const int rc = prepare_stream_pipeline(argv[1], argv[2], argv[3], hardware, textured,
                                                adapter, device, pso)) return rc;
    // <vs> <ps> <root> slot67 <code> <data> <state> [case]: WARP device and root only, then slot-67 creations.
    if (argc == 9 && std::strcmp(argv[4], "slot67") == 0 && std::strncmp(argv[8], "dispatch", 8) == 0) {
        if (std::strncmp(argv[8], "dispatch-heap-dynamic", 21) == 0) {
            ComPtr<ID3D12Device> desktop;
            D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
            D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
            if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&desktop))) ||
                FAILED(desktop->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))) ||
                FAILED(desktop->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
                model.HighestShaderModel < D3D_SHADER_MODEL_6_6 ||
                options.ResourceBindingTier < D3D12_RESOURCE_BINDING_TIER_3) {
                std::puts("direct heap indexing requires Shader Model 6.6 and resource binding tier 3");
                return 77;
            }
        }
        if (!create_slot67(device, argv[5], argv[6], argv[7], argv[8])) return 10;
        const bool nolock_retry = std::strcmp(argv[8], "dispatch-nolock-retry") == 0;
        if (nolock_retry || std::strcmp(argv[8], "dispatch-retry") == 0) {
            if (int rc = dispatch_slot67(adapter.Get(), device, nolock_retry ? "dispatch-nolock-burst" : "dispatch")) return rc;
            const auto before = nolock_retry ? compute_refusal_stats().background_reads : 0;
            std::puts("first slot 67 Dispatch fenced");
            std::fflush(stdout);
            if (std::getchar() != '\n') return 2;
            if (nolock_retry) {
                const ULONGLONG deadline = GetTickCount64() + 10000;
                while (compute_refusal_stats().background_reads == before) {
                    if (GetTickCount64() >= deadline) return 84;
                    Sleep(10);
                }
            }
        }
        return dispatch_slot67(adapter.Get(), device, argv[8]);
    }
    if (argc == 9 && std::strcmp(argv[4], "slot67") == 0 && std::strncmp(argv[8], "texture", 7) == 0)
        return create_slot67(device, argv[5], argv[6], argv[7], argv[8]) ? texture_slot67(device, argv[8]) : 10;
    if ((argc == 8 || argc == 9) && std::strcmp(argv[4], "slot67") == 0)
        return create_slot67(device, argv[5], argv[6], argv[7], argc == 9 ? argv[8] : "valid") ? 0 : 10;
    if (argc == 5 && std::strncmp(argv[4], "present-", 8) == 0) return present_gray(device, argv[4]);
    if (GetEnvironmentVariableW(L"X2W_TEST_TRUNCATED_ONLY", nullptr, 0)) return 0;
    if (argc == 4) return 0;
    DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
    UINT export_word = 0;
    bool two_targets = false;
    if (argc == 7 && std::strncmp(argv[6], "indirect-", 9) == 0) {
        // A pipeline used only through slot 61. slot57-draw: three refused Draw calls; slot57-bound: bound, no call;
        // slot57-dispatch: one refused Dispatch call; desktop-draw: a desktop pipeline and one refused Draw call;
        // slot57-unbound-draw: three direct Draws with no render target bound, then the not-drawn key report on stdout.
        const std::string mode(argv[6] + 9);
        if (mode == "desktop-draw") pso = desktop_pipeline(device);
        else if (mode == "slot57-draw" || mode == "slot57-bound" || mode == "slot57-dispatch" || mode == "slot57-unbound-draw") {
            if (!create_slot57(device, argv[1], argv[2], argv[3], "", pso)) return 10;
        } else return 2;
        if (!pso) return 10;
        const bool direct = mode == "slot57-unbound-draw";
        const int calls = mode == "slot57-draw" || direct ? 3 : mode == "slot57-bound" ? 0 : 1;
        if (!bind_and_execute_indirect(device, pso, mode == "slot57-dispatch", calls, direct)) return 11;
        if (direct) {
            // The refused draws by pipeline key, as the periodic report prints them.
            std::vector<char> report(x2w_d3d12_not_drawn_keys_report(nullptr, 0) + 1);
            x2w_d3d12_not_drawn_keys_report(report.data(), report.size());
            std::fputs(report.data(), stdout);
        }
        return 0;
    }
    if (argc == 8 && std::strncmp(argv[6], "slot57", 6) == 0) {
        // The same shaders and root through device slot 57, with the given words in state block A.
        // "slot57-mrt" binds a second RGBA8 target.
        two_targets = std::strcmp(argv[6], "slot57-mrt") == 0;
        if (!two_targets && std::strcmp(argv[6], "slot57") != 0) return 2;
        if (!create_slot57(device, argv[1], argv[2], argv[3], argv[7], pso)) return 10;
    } else if (argc == 7 && !std::isdigit(static_cast<unsigned char>(argv[6][0]))) {
        const std::string mode(argv[6]);
        if (mode != "parent-a8" && mode != "child-a8" && mode != "child-r8") return 2;
        if (!derive(device, pso, mode != "parent-a8")) return 10;
        format = mode == "child-r8" ? DXGI_FORMAT_R8_UNORM : DXGI_FORMAT_A8_UNORM;
    } else if (argc == 7 || argc == 8) {
        // A render-target format for the translated MRT export, and an optional root constant.
        format = static_cast<DXGI_FORMAT>(std::strtoul(argv[6], nullptr, 10));
        StreamMrtFormat mrt{};
        if (!stream_mrt_format(format, mrt)) return 2;
        if (argc == 8) export_word = static_cast<UINT>(std::strtoul(argv[7], nullptr, 0));
    }
    wchar_t manifest[MAX_PATH];
    const DWORD length = GetEnvironmentVariableW(L"X2W_TEST_PUBLISH_MANIFEST", manifest, MAX_PATH);
    if (length && length < MAX_PATH) {
        const std::string first = std::string(argv[5]) + ".first";
        if (const int rc = check_stream_draw(adapter.Get(), device, pso, first.c_str(), nullptr, format)) return rc;
        if (!MoveFileW((std::wstring(manifest) + L".pending").c_str(), manifest)) return 61;
        Sleep(1100);
    }
    if (GetEnvironmentVariableW(L"X2W_TEST_PLACED_VERTEX", nullptr, 0)) {
        extern void release_placed_vertices(bool);
        const std::string first = std::string(argv[5]) + ".first";
        if (const int rc = check_stream_draw(adapter.Get(), device, pso, first.c_str(), nullptr, format)) return rc;
        // GPU fence and readback are complete before the high range is released.
        release_placed_vertices(false);
        const int rc = check_stream_draw(adapter.Get(), device, pso, argv[5], nullptr, format);
        release_placed_vertices(true);
        return rc;
    }
    return check_stream_draw(adapter.Get(), device, pso, argv[5], bc7.blocks ? &bc7 : nullptr, format, export_word, two_targets);
}
