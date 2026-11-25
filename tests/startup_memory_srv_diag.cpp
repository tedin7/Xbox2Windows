// Census of refused device slot 70 calls and of "token" texture nulls (src/startup_stubs/memory_srv_census.h).
//
// Eight refusals with a zero address spend the slot 70 line limit. Then a 2048x2048 BC1 texture with
// 10 mips is refused (unsupported_description): its line must still appear, as the first distinct
// shape. It leaves the descriptor-table entry untouched, so the draw in column 0 samples an entry of
// zeros and binds a null view ("token"). A supported 1-mip R8 texture then fills the entry, and the
// draw in column 1 samples it. With X2W_TEST_MEMORY_SRV_DIAG_CONTROL the refused call is replaced by
// that supported R8 publish: the control passes only on the mismatch that causes in column 0.
#include <d3d12.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

using Word = std::uint64_t;
Word invoke(void*, unsigned, Word = 0, Word = 0, Word = 0, Word = 0, Word = 0, Word = 0, Word = 0, Word = 0);
extern "C" __declspec(dllimport) std::size_t x2w_d3d12_census_report(char* out, std::size_t capacity);

namespace {
template <class T> Word ptr(T* value) { return reinterpret_cast<Word>(value); }
struct ViewAt {
    D3D12_SHADER_RESOURCE_VIEW_DESC view;
    Word address;
    Word tail[2];
};
static_assert(offsetof(ViewAt, address) == 0x28 && sizeof(ViewAt) == 0x40);
}  // namespace

int draw_memory_srv_diagnostics(ID3D12Device* desktop, void* guest, void* list)
{
    const bool control = GetEnvironmentVariableW(L"X2W_TEST_MEMORY_SRV_DIAG_CONTROL", nullptr, 0) != 0;
    const D3D12_RESOURCE_DESC r8{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 2, 2, 1, 1,
        DXGI_FORMAT_R8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    const D3D12_RESOURCE_DESC bc1{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 2048, 2048, 1, 10,
        DXGI_FORMAT_BC1_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, bytes = 0;
    desktop->GetCopyableFootprints(&r8, 0, 1, 0, &footprint, &rows, &row_bytes, &bytes);
    if (rows != 2 || row_bytes != 2) return 50;
    static std::vector<unsigned char> plane;
    plane.assign(static_cast<std::size_t>(bytes), 0);
    for (unsigned y = 0; y < rows; ++y)
        for (unsigned x = 0; x < row_bytes; ++x) plane[y * footprint.Footprint.RowPitch + x] = static_cast<unsigned char>(16 + 32 * y + 7 * x);
    void* placed = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(plane.data()), ptr(&r8), D3D12_RESOURCE_STATE_COMMON, 0,
                                           ptr(&__uuidof(ID3D12Resource)), ptr(&placed))))) return 51;

    struct Extended {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word address, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
    void* heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr(&heap_desc), ptr(&__uuidof(ID3D12DescriptorHeap)), ptr(&heap))))) return 52;
    const Word cpu = invoke(heap, 9);
    invoke(list, 32, 1, ptr(&heap));
    invoke(list, 36, 0, invoke(heap, 10));

    ViewAt v{};
    v.view.Format = DXGI_FORMAT_R8_UNORM;
    v.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    v.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.view.Texture2D.MipLevels = 1;
    for (int i = 0; i < 8; ++i) invoke(guest, 70, ptr(&r8), ptr(&v), cpu);  // null_address: spends the line limit
    ViewAt refused = v;
    refused.address = ptr(plane.data());
    if (!control) {
        refused.view.Format = DXGI_FORMAT_BC1_UNORM;
        refused.view.Texture2D.MipLevels = 10;
    }
    invoke(guest, 70, ptr(control ? &r8 : &bc1), ptr(&refused), cpu);
    const auto draw = [&](LONG x) {
        const D3D12_RECT strip{x, 0, x + 1, 4};
        invoke(list, 22, 1, ptr(&strip));
        invoke(list, 12, 3, 1);
    };
    draw(0);
    v.address = ptr(plane.data());
    invoke(guest, 70, ptr(&r8), ptr(&v), cpu);
    draw(1);

    std::array<char, 4096> report{};
    x2w_d3d12_census_report(report.data(), report.size());
    std::printf("census report begin\n%scensus report end\n", report.data());
    std::fflush(stdout);
    return 0;
}
