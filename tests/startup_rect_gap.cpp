// X2W_TEST_RECT_GAP (#3131): the eight-vertex quad-and-edge-ring draw.
//
// Every vertex record is 12 bytes: a binary32 mask at +0, then XY at +4. Two vertex-buffer views name
// the same records, slot 0 (TEXCOORD, R32_FLOAT at offset 0, the mask) and slot 1 (POSITION,
// R32G32_FLOAT at offset 4), both with a 12-byte stride. Records 0-2 are the edge ring that encloses the
// quad (mask 0) and 3-5 the inner quad (mask 1); 6 and 7 are their fourth corners, so a rectangle list of
// six vertices and an indexed triangle list built from the same bytes describe the same four triangles.
// The ring is drawn first, so the quad's diagonal stays visible. A mask-controlled pixel shader paints
// the quad red and the ring green; a constant-colour pixel shader paints whatever the geometry covers red.
//
//   triangle        R16 indexed triangle list, ring then quad (12 indices: 0-1-2, 0-2-6, 3-4-5, 3-5-7)
//   triangle-copy   the same draw from a vertex stream and an index view that no placed buffer holds,
//                   so the recording-time vertex read and the index scan size the copy
//   rect            non-indexed rectangle list of six vertices (the ring and the quad)
//   rect8           the same rectangle list with eight vertices, two of them unused
//   rect8-copy      the eight-vertex rectangle list from an unregistered vertex stream
//   none            no draw: the cleared target, read back through the same path
//
// X2W_TEST_RECT_GAP_BYTES names a file that receives the vertex bytes the draw bound, then the index
// bytes it bound, so the check can compare them with what it wrote.
#include <d3d12.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

#include "startup_stream_setup.h"

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }

struct Vertex {
    float mask, x, y;
};
// Each rectangle is three corners clockwise on screen (bottom-left, top-left, top-right); the fourth
// corner of a rectangle is v0 + v2 - v1 and is stored at 6 and 7.
constexpr std::array<Vertex, 8> vertices{{
    {0.0f, -1.0f, -1.0f}, {0.0f, -1.0f, 1.0f}, {0.0f, 1.0f, 1.0f},              // ring
    {1.0f, -0.75f, -0.625f}, {1.0f, -0.75f, 0.625f}, {1.0f, 0.75f, 0.625f},     // quad
    {0.0f, 1.0f, -1.0f}, {1.0f, 0.75f, -0.625f},                                // the fourth corners
}};
constexpr std::array<std::uint16_t, 12> indices{{0, 1, 2, 0, 2, 6, 3, 4, 5, 3, 5, 7}};

// Memory no placed buffer covers: the draw then copies the bytes it needs while recording.
alignas(64) unsigned char unplaced[256];
}

int draw_rect_gap(void* guest, void* list)
{
    char mode[32]{};
    if (!GetEnvironmentVariableA("X2W_TEST_RECT_GAP", mode, sizeof(mode))) return 2;
    // The stream harness scissors an 8x4 target to its left half by default; this draw covers the whole
    // target, so both sides of each diagonal are read back.
    const D3D12_RECT whole{0, 0, 8, 4};
    invoke(list, 22, 1, arg(&whole));
    if (std::strcmp(mode, "none") == 0) return 0;
    const bool indexed = std::strcmp(mode, "triangle") == 0 || std::strcmp(mode, "triangle-copy") == 0;
    const bool short_view = std::strcmp(mode, "triangle-short") == 0;
    const bool copied = std::strstr(mode, "-copy") != nullptr;
    const bool rotated = std::strstr(mode, "rotated") != nullptr;
    const bool indexed_path = indexed || short_view || std::strcmp(mode, "triangle-rotated") == 0;
    if (!indexed_path && std::strncmp(mode, "rect", 4) != 0) return 2;

    // A rotation keeps the parallelogram, so v0 + v2 - v1 is still its fourth corner, but no corner shares
    // an axis with its two neighbours any more.
    std::array<Vertex, 8> records = vertices;
    if (rotated) {
        const float angle = 0.2617993877991494f;  // 15 degrees
        const float c = std::cos(angle), s = std::sin(angle);
        for (Vertex& v : records) {
            const float x = v.x, y = v.y;
            v.x = c * x - s * y;
            v.y = s * x + c * y;
        }
    }

    void* const placed = VirtualAlloc(nullptr, 0x10000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!placed) return 43;
    unsigned char* const vertex_bytes = copied ? unplaced : static_cast<unsigned char*>(placed);
    unsigned char* const index_bytes = static_cast<unsigned char*>(placed) + 0x2000;
    std::memcpy(vertex_bytes, records.data(), sizeof(records));
    std::memcpy(index_bytes, indices.data(), sizeof(indices));
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 0x10000;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    void* resource = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 133, arg(static_cast<unsigned char*>(placed)), arg(&desc),
            D3D12_RESOURCE_STATE_COMMON, 0, arg(&__uuidof(ID3D12Resource)), arg(&resource))))) return 44;
    // triangle-short declares a three-record view although its indices name records up to seven: desktop IA
    // reads the bytes the indices name, so the quad still has to be covered.
    const UINT view_bytes = short_view ? 3 * sizeof(Vertex) : static_cast<UINT>(sizeof(records));
    const D3D12_VERTEX_BUFFER_VIEW views[2] = {
        {arg(vertex_bytes), view_bytes, sizeof(Vertex)}, {arg(vertex_bytes), view_bytes, sizeof(Vertex)}};
    invoke(list, 48, 0, 2, arg(views));
    if (indexed_path) {
        // The quad's indices, 3-4-5 and 3-5-7, follow the ring's six.
        const UINT index_offset = short_view ? 12 : 0;
        const D3D12_INDEX_BUFFER_VIEW index_view{arg(index_bytes + index_offset),
                                                 static_cast<UINT>(sizeof(indices)) - index_offset,
                                                 DXGI_FORMAT_R16_UINT};
        invoke(list, 47, arg(&index_view));
        invoke(list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        invoke(list, 13, static_cast<Word>(std::size(indices)) - index_offset / 2, 1, 0, 0, 0);
    } else {
        invoke(list, 20, 17 /* D3D_PRIMITIVE_TOPOLOGY_RECTLIST, the console topology value */);
        invoke(list, 12, std::strncmp(mode, "rect8", 5) == 0 ? 8 : 6, 1);
    }
    char dump[MAX_PATH]{};
    const DWORD length = GetEnvironmentVariableA("X2W_TEST_RECT_GAP_BYTES", dump, MAX_PATH);
    if (length && length < MAX_PATH) {
        std::ofstream file(dump, std::ios::binary);
        file.write(reinterpret_cast<const char*>(vertex_bytes), sizeof(records));
        if (indexed_path) file.write(reinterpret_cast<const char*>(index_bytes), sizeof(indices));
    }
    return 0;
}
