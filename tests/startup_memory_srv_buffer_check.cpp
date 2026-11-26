// Checks validation, offset/element adjustment, and GPU virtual address resolution for
// buffer memory SRVs (device slot 70) without a device.
// --mutate height: expect a buffer description with Height 2 to be accepted.
// --mutate exact-fit: expect a view that ends exactly at the end of the buffer to be refused.
// --mutate first-32: expect a resolved FirstElement above 4 Gi elements to be cut to 32 bits.
// --mutate unplaced-extent: expect page span calculation crossing a 64 KiB boundary to return a single page.
// --mutate unplaced-covers: expect a buffer range past the end of the cached page to be accepted as covered.
// --mutate unplaced-heap-limit: expect a 2 GiB allocation to be wrapped as an existing heap for an unplaced buffer.
// --mutate exact-start: expect a page-start base to be kept for a view address that is not a whole number of elements in.
// Each mutated run is a negative control and must fail.
#include "startup_stubs/memory_srv_buffer.h"

#include <cstdio>
#include <cstring>

namespace {

int failures;
const char* mutation = "";

bool mutated(const char* name)
{
    return std::strcmp(mutation, name) == 0;
}

void check(const char* name, bool condition)
{
    if (condition) return;
    ++failures;
    std::printf("FAIL %s\n", name);
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 width)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = width;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

D3D12_SHADER_RESOURCE_VIEW_DESC raw_view(UINT64 first_element, UINT num_elements)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = DXGI_FORMAT_R32_TYPELESS;
    v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Buffer.FirstElement = first_element;
    v.Buffer.NumElements = num_elements;
    v.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    return v;
}

D3D12_SHADER_RESOURCE_VIEW_DESC structured_view(UINT64 first_element, UINT num_elements, UINT stride)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = DXGI_FORMAT_UNKNOWN;
    v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Buffer.FirstElement = first_element;
    v.Buffer.NumElements = num_elements;
    v.Buffer.StructureByteStride = stride;
    return v;
}

D3D12_SHADER_RESOURCE_VIEW_DESC typed_view(DXGI_FORMAT format, UINT64 first_element, UINT num_elements)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = format;
    v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Buffer.FirstElement = first_element;
    v.Buffer.NumElements = num_elements;
    return v;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0 &&
        (std::strcmp(argv[2], "height") == 0 || std::strcmp(argv[2], "exact-fit") == 0 ||
         std::strcmp(argv[2], "first-32") == 0 ||
         std::strcmp(argv[2], "unplaced-extent") == 0 || std::strcmp(argv[2], "unplaced-covers") == 0 ||
         std::strcmp(argv[2], "unplaced-heap-limit") == 0 || std::strcmp(argv[2], "exact-start") == 0)) {
        mutation = argv[2];
    } else if (argc != 1) {
        std::printf("usage: startup_memory_srv_buffer_check [--mutate height|exact-fit|first-32|unplaced-extent|unplaced-covers|unplaced-heap-limit|exact-start]\n");
        return 2;
    }
    const auto buf = buffer_desc(1024);

    // Raw buffer SRV validation and interior offset resolution.
    const auto raw_ok = raw_view(4, 16);
    check("raw valid", x2w_startup::supports_buffer_memory_srv(buf, raw_ok));
    auto raw_bad_fmt = raw_ok;
    raw_bad_fmt.Format = DXGI_FORMAT_R32_FLOAT;
    check("raw bad format", !x2w_startup::supports_buffer_memory_srv(buf, raw_bad_fmt));
    auto raw_bad_stride = raw_ok;
    raw_bad_stride.Buffer.StructureByteStride = 4;
    check("raw nonzero stride", !x2w_startup::supports_buffer_memory_srv(buf, raw_bad_stride));
    check("raw zero count", !x2w_startup::supports_buffer_memory_srv(buf, raw_view(0, 0)));
    check("raw oob", !x2w_startup::supports_buffer_memory_srv(buf, raw_view(250, 10)));

    D3D12_SHADER_RESOURCE_VIEW_DESC resolved{};
    check("raw resolve base", x2w_startup::resolve_buffer_srv_desc(0x10000, 1024, 0x10000, raw_ok, resolved) &&
                                  resolved.Buffer.FirstElement == 4 && resolved.Buffer.NumElements == 16);
    check("raw resolve interior 16B",
          x2w_startup::resolve_buffer_srv_desc(0x10000, 1024, 0x10010, raw_ok, resolved) &&
              resolved.Buffer.FirstElement == 8);
    check("raw resolve unaligned 4B",
          !x2w_startup::resolve_buffer_srv_desc(0x10000, 1024, 0x10004, raw_view(0, 4), resolved));
    check("raw resolve oob",
          !x2w_startup::resolve_buffer_srv_desc(0x10000, 1024, 0x103e0, raw_view(0, 16), resolved));

    // Structured buffer SRV validation and interior offset resolution.
    const auto str_ok = structured_view(2, 8, 16);
    check("structured valid", x2w_startup::supports_buffer_memory_srv(buf, str_ok));
    auto str_bad_fmt = str_ok;
    str_bad_fmt.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    check("structured bad format", !x2w_startup::supports_buffer_memory_srv(buf, str_bad_fmt));
    check("structured stride not multiple of 4", !x2w_startup::supports_buffer_memory_srv(buf, structured_view(0, 1, 6)));
    check("structured stride 4 ok", x2w_startup::supports_buffer_memory_srv(buf, structured_view(0, 1, 4)));
    check("structured stride too large", !x2w_startup::supports_buffer_memory_srv(buf, structured_view(0, 1, 2052)));
    check("structured resolve interior",
          x2w_startup::resolve_buffer_srv_desc(0x10000, 1024, 0x10020, str_ok, resolved) &&
              resolved.Buffer.FirstElement == 4);
    check("structured resolve unaligned",
          !x2w_startup::resolve_buffer_srv_desc(0x10000, 1024, 0x10008, str_ok, resolved));

    // Typed buffer SRV validation and interior offset resolution.
    const auto typ_ok = typed_view(DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 4);
    check("typed valid", x2w_startup::supports_buffer_memory_srv(buf, typ_ok));
    check("typed unknown format", !x2w_startup::supports_buffer_memory_srv(buf, typed_view(DXGI_FORMAT_UNKNOWN, 0, 1)));
    check("typed element bytes 16", x2w_startup::typed_buffer_format_bytes(DXGI_FORMAT_R32G32B32A32_FLOAT) == 16);
    check("typed element bytes 12", x2w_startup::typed_buffer_format_bytes(DXGI_FORMAT_R32G32B32_UINT) == 12);
    check("typed element bytes 2", x2w_startup::typed_buffer_format_bytes(DXGI_FORMAT_R16_UNORM) == 2);
    check("typed srgb rejected", !x2w_startup::supports_buffer_memory_srv(buf, typed_view(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 0, 1)));
    check("typed bgra srgb rejected", !x2w_startup::supports_buffer_memory_srv(buf, typed_view(DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, 0, 1)));
    check("typed bc rejected", !x2w_startup::supports_buffer_memory_srv(buf, typed_view(DXGI_FORMAT_BC7_UNORM, 0, 1)));
    check("typed depth rejected", !x2w_startup::supports_buffer_memory_srv(buf, typed_view(DXGI_FORMAT_D32_FLOAT, 0, 1)));
    check("typed typeless rejected", !x2w_startup::supports_buffer_memory_srv(buf, typed_view(DXGI_FORMAT_R32G32B32A32_TYPELESS, 0, 1)));
    check("typed resolve interior",
          x2w_startup::resolve_buffer_srv_desc(0x10000, 1024, 0x10010, typ_ok, resolved) &&
              resolved.Buffer.FirstElement == 2);
    check("typed resolve unaligned",
          !x2w_startup::resolve_buffer_srv_desc(0x10000, 1024, 0x10004, typ_ok, resolved));

    // Denied shader resource flag on buffer resource.
    auto denied_buf = buf;
    denied_buf.Flags = D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
    check("denied srv flag", !x2w_startup::supports_buffer_memory_srv(denied_buf, raw_ok));

    // Each other buffer description field, changed alone from the accepted 1024-byte buffer.
    auto tall = buf;
    tall.Height = 2;
    check("resource height 2", x2w_startup::supports_buffer_memory_srv(tall, raw_ok) == mutated("height"));
    auto deep = buf;
    deep.DepthOrArraySize = 2;
    check("resource depth 2", !x2w_startup::supports_buffer_memory_srv(deep, raw_ok));
    auto mips = buf;
    mips.MipLevels = 2;
    check("resource mips 2", !x2w_startup::supports_buffer_memory_srv(mips, raw_ok));
    auto formatted = buf;
    formatted.Format = DXGI_FORMAT_R32_TYPELESS;
    check("resource format R32_TYPELESS", !x2w_startup::supports_buffer_memory_srv(formatted, raw_ok));
    auto multisampled = buf;
    multisampled.SampleDesc.Count = 4;
    check("resource 4 samples", !x2w_startup::supports_buffer_memory_srv(multisampled, raw_ok));
    auto quality = buf;
    quality.SampleDesc.Quality = 1;
    check("resource sample quality 1", !x2w_startup::supports_buffer_memory_srv(quality, raw_ok));
    auto unknown_layout = buf;
    unknown_layout.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    check("resource layout unknown", !x2w_startup::supports_buffer_memory_srv(unknown_layout, raw_ok));
    auto placed = buf;
    placed.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    check("resource alignment 64 KiB", x2w_startup::supports_buffer_memory_srv(placed, raw_ok));
    auto small_aligned = buf;
    small_aligned.Alignment = 4096;
    check("resource alignment 4 KiB", !x2w_startup::supports_buffer_memory_srv(small_aligned, raw_ok));

    // View fields outside the element size: dimension, component mapping and unknown flags.
    auto texture_dim = raw_ok;
    texture_dim.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    check("view dimension texture2d", !x2w_startup::supports_buffer_memory_srv(buf, texture_dim));
    auto swizzled = typ_ok;
    swizzled.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(3, 2, 1, 0);
    check("view swizzled mapping", !x2w_startup::supports_buffer_memory_srv(buf, swizzled));
    auto typed_flag = typ_ok;
    typed_flag.Buffer.Flags = static_cast<D3D12_BUFFER_SRV_FLAGS>(2);
    check("typed view unknown flag", !x2w_startup::supports_buffer_memory_srv(buf, typed_flag));
    auto raw_flag = raw_ok;
    raw_flag.Buffer.Flags = static_cast<D3D12_BUFFER_SRV_FLAGS>(D3D12_BUFFER_SRV_FLAG_RAW | 2);
    check("raw view unknown flag", !x2w_startup::supports_buffer_memory_srv(buf, raw_flag));
    check("structured stride 2048 bytes", x2w_startup::buffer_srv_element_bytes(structured_view(0, 1, 2048)) == 2048);
    check("structured stride 2048 ok",
          x2w_startup::supports_buffer_memory_srv(buffer_desc(4096), structured_view(0, 1, 2048)));

    // Element bytes of every typed buffer format, from its bits per element.
    struct FormatBytes {
        DXGI_FORMAT format;
        std::uint32_t bytes;
    };
    const FormatBytes format_bytes[] = {
        {DXGI_FORMAT_R32G32B32A32_FLOAT, 16}, {DXGI_FORMAT_R32G32B32A32_UINT, 16}, {DXGI_FORMAT_R32G32B32A32_SINT, 16},
        {DXGI_FORMAT_R32G32B32_FLOAT, 12},    {DXGI_FORMAT_R32G32B32_UINT, 12},    {DXGI_FORMAT_R32G32B32_SINT, 12},
        {DXGI_FORMAT_R16G16B16A16_FLOAT, 8},  {DXGI_FORMAT_R16G16B16A16_UNORM, 8}, {DXGI_FORMAT_R16G16B16A16_UINT, 8},
        {DXGI_FORMAT_R16G16B16A16_SNORM, 8},  {DXGI_FORMAT_R16G16B16A16_SINT, 8},  {DXGI_FORMAT_R32G32_FLOAT, 8},
        {DXGI_FORMAT_R32G32_UINT, 8},         {DXGI_FORMAT_R32G32_SINT, 8},        {DXGI_FORMAT_R10G10B10A2_UNORM, 4},
        {DXGI_FORMAT_R10G10B10A2_UINT, 4},    {DXGI_FORMAT_R11G11B10_FLOAT, 4},    {DXGI_FORMAT_R8G8B8A8_UNORM, 4},
        {DXGI_FORMAT_R8G8B8A8_UINT, 4},       {DXGI_FORMAT_R8G8B8A8_SNORM, 4},     {DXGI_FORMAT_R8G8B8A8_SINT, 4},
        {DXGI_FORMAT_R16G16_FLOAT, 4},        {DXGI_FORMAT_R16G16_UNORM, 4},       {DXGI_FORMAT_R16G16_UINT, 4},
        {DXGI_FORMAT_R16G16_SNORM, 4},        {DXGI_FORMAT_R16G16_SINT, 4},        {DXGI_FORMAT_R32_FLOAT, 4},
        {DXGI_FORMAT_R32_UINT, 4},            {DXGI_FORMAT_R32_SINT, 4},           {DXGI_FORMAT_R8G8_UNORM, 2},
        {DXGI_FORMAT_R8G8_UINT, 2},           {DXGI_FORMAT_R8G8_SNORM, 2},         {DXGI_FORMAT_R8G8_SINT, 2},
        {DXGI_FORMAT_R16_FLOAT, 2},           {DXGI_FORMAT_R16_UNORM, 2},          {DXGI_FORMAT_R16_UINT, 2},
        {DXGI_FORMAT_R16_SNORM, 2},           {DXGI_FORMAT_R16_SINT, 2},           {DXGI_FORMAT_R8_UNORM, 1},
        {DXGI_FORMAT_R8_UINT, 1},             {DXGI_FORMAT_R8_SNORM, 1},           {DXGI_FORMAT_R8_SINT, 1},
    };
    for (const auto& f : format_bytes) {
        if (x2w_startup::typed_buffer_format_bytes(f.format) == f.bytes) continue;
        ++failures;
        std::printf("FAIL typed element bytes of format %d: got %u, expected %u\n", static_cast<int>(f.format),
                    x2w_startup::typed_buffer_format_bytes(f.format), f.bytes);
    }

    // A view that ends exactly at the end of the buffer is accepted; one element more is refused.
    check("raw exact fit", x2w_startup::supports_buffer_memory_srv(buf, raw_view(0, 256)) == !mutated("exact-fit"));
    check("raw one past the end", !x2w_startup::supports_buffer_memory_srv(buf, raw_view(1, 256)));
    check("raw first at the end", !x2w_startup::supports_buffer_memory_srv(buf, raw_view(256, 1)));
    check("typed R8 exact fit",
          x2w_startup::supports_buffer_memory_srv(buf, typed_view(DXGI_FORMAT_R8_UINT, 0, 1024)) == !mutated("exact-fit"));
    check("typed R8 one past the end",
          !x2w_startup::supports_buffer_memory_srv(buf, typed_view(DXGI_FORMAT_R8_UINT, 0, 1025)));
    // Address 0x10010: 16 bytes in, 252 raw elements left.
    check("raw resolve exact fit",
          x2w_startup::resolve_buffer_srv_desc(0x10000, 1024, 0x10010, raw_view(4, 248), resolved) ==
                  !mutated("exact-fit") &&
              (mutated("exact-fit") || (resolved.Buffer.FirstElement == 8 && resolved.Buffer.NumElements == 248)));
    check("raw resolve one past the end",
          !x2w_startup::resolve_buffer_srv_desc(0x10000, 1024, 0x10010, raw_view(4, 249), resolved));
    // Address 0x10020: 2 of 64 structured elements in, 62 left, so FirstElement 63 is past the end.
    check("structured resolve first past the end",
          !x2w_startup::resolve_buffer_srv_desc(0x10000, 1024, 0x10020, structured_view(63, 1, 16), resolved));

    // Raw alignment is of the adjusted first element, not of the address offset alone.
    check("raw resolve offset 12 first 1",
          x2w_startup::resolve_buffer_srv_desc(0x10000, 1024, 0x1000c, raw_view(1, 4), resolved) &&
              resolved.Buffer.FirstElement == 4);
    check("raw resolve offset 16 first 1",
          !x2w_startup::resolve_buffer_srv_desc(0x10000, 1024, 0x10010, raw_view(1, 4), resolved));

    // Widths, addresses and element indices above 4 GiB keep all 64 bits.
    const auto wide = buffer_desc(0x100000400);  // 0x40000100 raw elements
    check("raw 64-bit width", x2w_startup::supports_buffer_memory_srv(wide, raw_view(0x40000000, 0x40)));
    check("raw 64-bit width one past the end", !x2w_startup::supports_buffer_memory_srv(wide, raw_view(0x40000000, 0x101)));
    const std::uint64_t first_32 = mutated("first-32") ? 0x10 : 0x100000010;
    check("typed R8 resolve 4 GiB in",
          x2w_startup::resolve_buffer_srv_desc(0x200000000, 0x200000000, 0x300000000,
                                               typed_view(DXGI_FORMAT_R8_UINT, 0x10, 0x100), resolved) &&
              resolved.Buffer.FirstElement == first_32 && resolved.Buffer.NumElements == 0x100);
    check("typed R8 resolve 64-bit first element",
          x2w_startup::resolve_buffer_srv_desc(0x200000000, 0x200000000, 0x200000000,
                                               typed_view(DXGI_FORMAT_R8_UINT, 0x100000000, 0x10), resolved) &&
              resolved.Buffer.FirstElement == (mutated("first-32") ? 0 : 0x100000000));

    // Unplaced buffer extent calculation and page alignment.
    std::uint64_t page_start = 0, page_span = 0;
    check("unplaced extent zero address", !x2w_startup::unplaced_buffer_extent(0, 1024, page_start, page_span));
    check("unplaced extent zero span", !x2w_startup::unplaced_buffer_extent(0x10000, 0, page_start, page_span));
    check("unplaced extent 48B aligned",
          x2w_startup::unplaced_buffer_extent(0x10000, 48, page_start, page_span) &&
              page_start == 0x10000 && page_span == 65536);
    check("unplaced extent interior 128B",
          x2w_startup::unplaced_buffer_extent(0x10800, 128, page_start, page_span) &&
              page_start == 0x10000 && page_span == 65536);
    const std::uint64_t expected_cross_span = mutated("unplaced-extent") ? 65536 : 131072;
    check("unplaced extent crossing page",
          x2w_startup::unplaced_buffer_extent(0x1fff0, 32, page_start, page_span) &&
              page_start == 0x10000 && page_span == expected_cross_span);
    check("unplaced extent 128 KiB",
          x2w_startup::unplaced_buffer_extent(0x10000, 128 * 1024, page_start, page_span) &&
              page_start == 0x10000 && page_span == 128 * 1024);
    check("unplaced extent overflow address",
          !x2w_startup::unplaced_buffer_extent(~std::uint64_t{0} - 15, 32, page_start, page_span));
    check("unplaced extent overflow span",
          !x2w_startup::unplaced_buffer_extent(0x10000, ~std::uint64_t{0}, page_start, page_span));
    check("unplaced extent exceeds 256 MiB",
          !x2w_startup::unplaced_buffer_extent(0x10000, (256ull << 20) + 1, page_start, page_span));

    // Unplaced buffer cache coverage checks.
    check("unplaced covers interior", x2w_startup::unplaced_buffer_covers(0x10000, 65536, 0x10200, 128));
    check("unplaced covers start exact", x2w_startup::unplaced_buffer_covers(0x10000, 65536, 0x10000, 65536));
    check("unplaced covers end exact", x2w_startup::unplaced_buffer_covers(0x10000, 65536, 0x1ff00, 256));
    check("unplaced covers past end",
          x2w_startup::unplaced_buffer_covers(0x10000, 65536, 0x1ff00, 257) == mutated("unplaced-covers"));
    check("unplaced covers before start", !x2w_startup::unplaced_buffer_covers(0x10000, 65536, 0x0fff0, 32));
    check("unplaced covers zero cached span", !x2w_startup::unplaced_buffer_covers(0x10000, 0, 0x10000, 48));
    check("unplaced covers zero requested span", !x2w_startup::unplaced_buffer_covers(0x10000, 65536, 0x10000, 0));

    // Shapes of the three structured views the title published at sub-allocated addresses (low 16 bits of the address, stride).
    const std::uint64_t shapes[3][2] = {{0x6200, 2048}, {0x1c80, 320}, {0x6a00, 180}};
    for (const auto& shape : shapes) {
        const std::uint64_t address = 0x7f0000000000ull + shape[0];
        const std::uint64_t page = address & ~0xffffull;
        D3D12_SHADER_RESOURCE_VIEW_DESC structured{};
        structured.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        structured.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        structured.Buffer.NumElements = 1;
        structured.Buffer.StructureByteStride = static_cast<UINT>(shape[1]);
        D3D12_SHADER_RESOURCE_VIEW_DESC out{};
        check("page-start base cannot name a sub-allocated structured view",
              !x2w_startup::resolve_buffer_srv_desc(page, 65536, address, structured, out));
        const bool exact = x2w_startup::unplaced_view_needs_exact_start(address, page, shape[1]) && !mutated("exact-start");
        const std::uint64_t base = exact ? address : page;
        check("exact-start base names a sub-allocated structured view",
              x2w_startup::resolve_buffer_srv_desc(base, shape[1], address, structured, out) && out.Buffer.FirstElement == 0);
    }
    check("aligned address keeps the page base", !x2w_startup::unplaced_view_needs_exact_start(0x14000, 0x10000, 2048));

    // Existing-heap wrap of the allocation holding an unplaced buffer: up to the 256 MiB span limit only.
    check("unplaced heap 64 KiB allocation", x2w_startup::unplaced_existing_heap_allowed(65536));
    check("unplaced heap 256 MiB allocation", x2w_startup::unplaced_existing_heap_allowed(256ull << 20));
    check("unplaced heap 256 MiB + 64 KiB allocation",
          !x2w_startup::unplaced_existing_heap_allowed((256ull << 20) + 65536));
    check("unplaced heap 2 GiB allocation",
          x2w_startup::unplaced_existing_heap_allowed(2ull << 30) == mutated("unplaced-heap-limit"));
    check("unplaced heap unknown allocation size", !x2w_startup::unplaced_existing_heap_allowed(0));

    if (failures) return 1;
    std::printf("slot 70 buffer memory SRV validation: all checks passed\n");
    return 0;
}
