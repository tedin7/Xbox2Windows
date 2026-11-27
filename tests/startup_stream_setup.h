#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include <d3d12.h>
#include <dxgi.h>
#include <wrl/client.h>

int pick_stream_adapter(bool hardware, Microsoft::WRL::ComPtr<IDXGIAdapter>& adapter);
int prepare_stream_pipeline(const char* vs_path, const char* ps_path, const char* root_path,
                            bool hardware, bool textured, Microsoft::WRL::ComPtr<IDXGIAdapter>& adapter,
                            void*& device, void*& pso);

// The guest root signature that prepare_stream_pipeline created (null before it ran).
void* stream_root_signature();
void* stream_alternate_root_signature();
// The serialized form of stream_root_signature().
const std::vector<char>& stream_root_blob();

struct StreamAttachment {
    Microsoft::WRL::ComPtr<ID3D12Resource> texture, readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT row_bytes = 0;
};

struct StreamTarget {
    std::array<StreamAttachment, 2> attachments;
    UINT count{};
    DXGI_FORMAT second_format = DXGI_FORMAT_R8G8B8A8_UNORM;
    void* rtv_heap{};
    void* list{};
    void* queue{};
    void* allocator{};
    void* fence{};
};
int read_stream_target(ID3D12Device* desktop, StreamTarget& state, const char* pixels);
int open_stream_target(ID3D12Device* desktop, void* guest, void* pso, UINT width, UINT height,
                       DXGI_FORMAT format, UINT scissor_width, StreamTarget& state);
// Typed render-target formats the translated colour export covers (#1999 F). Integer targets are created
// typeless and drawn with a second RGBA8 target; float targets are drawn alone.
struct StreamMrtFormat {
    bool integer;
    bool single;
    UINT bytes;
    DXGI_FORMAT typeless;
};
inline bool stream_mrt_format(DXGI_FORMAT format, StreamMrtFormat& out)
{
    switch (format) {
    case DXGI_FORMAT_R8_UINT: case DXGI_FORMAT_R8_SINT:
        out = {true, true, 1, DXGI_FORMAT_R8_TYPELESS}; return true;
    case DXGI_FORMAT_R8G8_UINT: case DXGI_FORMAT_R8G8_SINT:
        out = {true, true, 2, DXGI_FORMAT_R8G8_TYPELESS}; return true;
    case DXGI_FORMAT_R16_UINT: case DXGI_FORMAT_R16_SINT:
        out = {true, true, 2, DXGI_FORMAT_R16_TYPELESS}; return true;
    case DXGI_FORMAT_R16G16_UINT: case DXGI_FORMAT_R16G16_SINT:
        out = {true, true, 4, DXGI_FORMAT_R16G16_TYPELESS}; return true;
    case DXGI_FORMAT_R16G16B16A16_UINT: case DXGI_FORMAT_R16G16B16A16_SINT:
        out = {true, true, 8, DXGI_FORMAT_R16G16B16A16_TYPELESS}; return true;
    case DXGI_FORMAT_R10G10B10A2_UINT:
        out = {true, true, 4, DXGI_FORMAT_R10G10B10A2_TYPELESS}; return true;
    case DXGI_FORMAT_R8G8B8A8_UINT: case DXGI_FORMAT_R8G8B8A8_SINT:
        out = {true, true, 4, DXGI_FORMAT_R8G8B8A8_TYPELESS}; return true;
    case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_SINT: out = {true, true, 4, DXGI_FORMAT_R32_TYPELESS}; return true;
    case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R32G32_SINT: out = {true, true, 8, DXGI_FORMAT_R32G32_TYPELESS}; return true;
    case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R32G32B32A32_SINT:
        out = {true, true, 16, DXGI_FORMAT_R32G32B32A32_TYPELESS}; return true;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: out = {false, true, 8, format}; return true;
    case DXGI_FORMAT_R32_FLOAT: out = {false, true, 4, format}; return true;
    case DXGI_FORMAT_R32G32_FLOAT: out = {false, true, 8, format}; return true;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: out = {false, true, 16, format}; return true;
    default: return false;
    }
}
using StreamWord = std::uint64_t;
StreamWord invoke(void*, unsigned, StreamWord = 0, StreamWord = 0, StreamWord = 0, StreamWord = 0,
                  StreamWord = 0, StreamWord = 0, StreamWord = 0, StreamWord = 0);
