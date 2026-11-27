// Checks src/startup_stubs/packed_target_format.h without a device, against adapters described
// by their format-support bits.
// --mutate stand-in | array | b565 | sharedexp | sharedexp-uav | view | tag: the helper's answer
// perturbed the way a wrong helper would answer (a stand-in in another format, arrays left alone as
// before, B5G6R5_UNORM left alone as before, R9G9B9E5_SHAREDEXP left alone as before, a UAV-only
// R9G9B9E5_SHAREDEXP texture left alone as under a render-target-only trigger, the view format of the
// stand-in kept packed, the stand-in's tag read back as no tag);
// each must fail on the rows it concerns. --mutate raw-clear forwards console depth format 118 unchanged.
#include "startup_stubs/clear_value_format.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <winerror.h>

namespace {
int failures, checks;
const char* mutation = "";
constexpr UINT rt = D3D12_FORMAT_SUPPORT1_RENDER_TARGET, uav = D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW;

struct Adapter {
    UINT b5g5r5a1, b4g4r4a4, rgba8, b5g6r5;
    UINT sharedexp = 0, hdr = 0;
    UINT operator()(DXGI_FORMAT format) const
    {
        if (format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP) return sharedexp;
        if (format == DXGI_FORMAT_R11G11B10_FLOAT) return hdr;
        return format == DXGI_FORMAT_B5G5R5A1_UNORM ? b5g5r5a1 : format == DXGI_FORMAT_B4G4R4A4_UNORM ? b4g4r4a4
             : format == DXGI_FORMAT_B5G6R5_UNORM ? b5g6r5 : format == DXGI_FORMAT_R8G8B8A8_UNORM ? rgba8 : rt | uav;
    }
};

// The helper under test, with the wrong answers a --mutate run substitutes.
DXGI_FORMAT stand_in(const D3D12_RESOURCE_DESC& desc, const Adapter& adapter)
{
    const DXGI_FORMAT got = x2w_startup::packed_target_stand_in(desc, adapter);
    if (!std::strcmp(mutation, "stand-in") && got == DXGI_FORMAT_R8G8B8A8_UNORM) return DXGI_FORMAT_B8G8R8A8_UNORM;
    if (!std::strcmp(mutation, "array") && desc.DepthOrArraySize != 1) return desc.Format;
    if (!std::strcmp(mutation, "b565") && desc.Format == DXGI_FORMAT_B5G6R5_UNORM) return desc.Format;
    if (!std::strcmp(mutation, "sharedexp") && desc.Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP) return desc.Format;
    if (!std::strcmp(mutation, "sharedexp-uav") && desc.Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP &&
        !(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET))
        return desc.Format;
    return got;
}

x2w_startup::PackedView packed_view(DXGI_FORMAT view_format, DXGI_FORMAT resource_format, DXGI_FORMAT stands_in_for)
{
    const auto got = x2w_startup::packed_view(view_format, resource_format, stands_in_for);
    if (!std::strcmp(mutation, "view") && got == x2w_startup::PackedView::stand_in) return x2w_startup::PackedView::unchanged;
    return got;
}

DXGI_FORMAT packed_stand_in_of(ID3D12Resource* resource)
{
    const DXGI_FORMAT got = x2w_startup::packed_stand_in_of(resource);
    if (!std::strcmp(mutation, "tag")) return DXGI_FORMAT_UNKNOWN;
    return got;
}

// A created texture as the tag helpers see it: its format and one private-data entry, which
// GetPrivateData answers the way a D3D12 object does (the stored size, DXGI_ERROR_NOT_FOUND for
// another GUID, DXGI_ERROR_MORE_DATA for a short buffer). Nothing else is called.
class Texture final : public ID3D12Resource {
public:
    explicit Texture(DXGI_FORMAT format) : format_(format) {}
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT* size, void* data) override
    {
        if (!size) return E_INVALIDARG;
        if (!stored_ || guid != guid_) {
            *size = 0;
            return DXGI_ERROR_NOT_FOUND;
        }
        if (!data) {
            *size = bytes_;
            return S_OK;
        }
        if (*size < bytes_) {
            *size = bytes_;
            return DXGI_ERROR_MORE_DATA;
        }
        std::memcpy(data, data_, bytes_);
        *size = bytes_;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT size, const void* data) override
    {
        if (size > sizeof(data_) || (size && !data)) return E_INVALIDARG;
        guid_ = guid;
        bytes_ = size;
        stored_ = true;
        if (size) std::memcpy(data_, data, size);
        return S_OK;
    }
    D3D12_RESOURCE_DESC STDMETHODCALLTYPE GetDesc() override
    {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = 64;
        desc.Height = 64;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = format_;
        desc.SampleDesc.Count = 1;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        return desc;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void** out) override
    {
        if (out) *out = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void** out) override
    {
        if (out) *out = nullptr;
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE Map(UINT, const D3D12_RANGE*, void**) override { return E_NOTIMPL; }
    void STDMETHODCALLTYPE Unmap(UINT, const D3D12_RANGE*) override {}
    D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE GetGPUVirtualAddress() override { return 0; }
    HRESULT STDMETHODCALLTYPE WriteToSubresource(UINT, const D3D12_BOX*, const void*, UINT, UINT) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE ReadFromSubresource(void*, UINT, UINT, UINT, const D3D12_BOX*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE GetHeapProperties(D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS*) override { return E_NOTIMPL; }

private:
    DXGI_FORMAT format_;
    GUID guid_{};
    UINT bytes_ = 0;
    bool stored_ = false;
    unsigned char data_[16]{};
};

D3D12_RESOURCE_DESC texture(DXGI_FORMAT format, UINT flags, D3D12_RESOURCE_DIMENSION dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = dimension;
    desc.Width = 2560;
    desc.Height = 1440;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = static_cast<D3D12_RESOURCE_FLAGS>(flags);
    return desc;
}

void expect(const char* name, DXGI_FORMAT got, DXGI_FORMAT wanted)
{
    ++checks;
    if (got != wanted) {
        std::printf("FAIL %s: format %d, expected %d\n", name, got, wanted);
        ++failures;
    }
}

void view(const char* name, x2w_startup::PackedView got, x2w_startup::PackedView wanted)
{
    ++checks;
    if (got != wanted) {
        std::printf("FAIL %s: verdict %d, expected %d\n", name, static_cast<int>(got), static_cast<int>(wanted));
        ++failures;
    }
}
void check_clear(const char* name, const D3D12_CLEAR_VALUE& clear, DXGI_FORMAT requested, DXGI_FORMAT created,
                 DXGI_FORMAT expected_format, bool rewritten)
{
    D3D12_CLEAR_VALUE original;
    std::memcpy(&original, &clear, sizeof(original));
    D3D12_CLEAR_VALUE copy{};
    const auto* forwarded = x2w_startup::desktop_clear_value(&clear, requested, created, copy);
    if (!std::strcmp(mutation, "raw-clear") && clear.Format == static_cast<DXGI_FORMAT>(118))
        forwarded = &clear;
    constexpr auto payload_offset = offsetof(D3D12_CLEAR_VALUE, Color);
    const auto payload = [](const D3D12_CLEAR_VALUE& value) {
        return reinterpret_cast<const unsigned char*>(&value) + payload_offset;
    };
    ++checks;
    if (forwarded != (rewritten ? &copy : &clear) || forwarded->Format != expected_format ||
        std::memcmp(payload(*forwarded), payload(original), sizeof(original.Color)) != 0 ||
        std::memcmp(&clear, &original, sizeof(original)) != 0) {
        std::printf("FAIL %s: incorrect clear format, copy, or payload\n", name);
        ++failures;
    }
}

void check_clear_values()
{
    D3D12_CLEAR_VALUE depth{};
    depth.Format = static_cast<DXGI_FORMAT>(118);
    depth.DepthStencil = {0.375f, 0xa5};
    check_clear("console depth/stencil", depth, depth.Format, DXGI_FORMAT_D24_UNORM_S8_UINT,
                DXGI_FORMAT_D24_UNORM_S8_UINT, true);

    depth.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    depth.DepthStencil = {0.0f, 0};
    check_clear("desktop zero depth", depth, depth.Format, depth.Format, depth.Format, false);

    D3D12_CLEAR_VALUE colour{DXGI_FORMAT_UNKNOWN, {0.125f, 0.25f, 0.5f, 1.0f}};
    for (const auto raw : {116, 117}) {
        colour.Format = static_cast<DXGI_FORMAT>(raw);
        check_clear("console 10-bit colour", colour, colour.Format, DXGI_FORMAT_R10G10B10A2_UNORM,
                    DXGI_FORMAT_R10G10B10A2_UNORM, true);
    }

    colour.Format = DXGI_FORMAT_B5G6R5_UNORM;
    check_clear("packed stand-in colour", colour, colour.Format, DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_R8G8B8A8_UNORM, true);
    check_clear("packed format retained", colour, colour.Format, colour.Format, colour.Format, false);
    check_clear("unrelated clear format", colour, DXGI_FORMAT_B4G4R4A4_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
                colour.Format, false);

    D3D12_CLEAR_VALUE copy{};
    ++checks;
    if (x2w_startup::desktop_clear_value(nullptr, depth.Format, depth.Format, copy)) {
        std::puts("FAIL null clear: expected nullptr");
        ++failures;
    }
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && !std::strcmp(argv[1], "--mutate")) mutation = argv[2];
    else if (argc != 1) {
        std::printf("usage: startup_packed_target_check [--mutate stand-in|array|b565|sharedexp|sharedexp-uav|view|tag|raw-clear]\n");
        return 2;
    }
    check_clear_values();
    const auto packed_target_stand_in = stand_in;
    constexpr UINT RT = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, UA = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    constexpr DXGI_FORMAT b5 = DXGI_FORMAT_B5G5R5A1_UNORM, b4 = DXGI_FORMAT_B4G4R4A4_UNORM, rgba = DXGI_FORMAT_R8G8B8A8_UNORM;
    constexpr DXGI_FORMAT b565 = DXGI_FORMAT_B5G6R5_UNORM;
    const Adapter full{rt | uav, rt | uav, rt | uav, rt | uav};   // supports the packed formats: nothing changes
    const Adapter none{0, 0, rt | uav, 0};                         // the measured case
    const Adapter render_only{rt, rt, rt | uav, rt};               // packed formats render but have no typed UAV
    const Adapter bare{0, 0, 0, 0};                                // no stand-in available either

    // The two descriptions the measured run was refused: render target with unordered access, and render target.
    expect("b5g5r5a1 RT|UAV, unsupported", packed_target_stand_in(texture(b5, RT | UA), none), rgba);
    expect("b4g4r4a4 RT, unsupported", packed_target_stand_in(texture(b4, RT), none), rgba);
    expect("b5g5r5a1 RT|UAV, render only", packed_target_stand_in(texture(b5, RT | UA), render_only), rgba);

    // Controls: the format stays when the adapter supports the use, when the stand-in would not
    // help, for other formats, and for every role outside the measured ones.
    expect("b5g5r5a1 RT|UAV, supported", packed_target_stand_in(texture(b5, RT | UA), full), b5);
    expect("b4g4r4a4 RT, supported", packed_target_stand_in(texture(b4, RT), full), b4);
    expect("b4g4r4a4 RT, render only", packed_target_stand_in(texture(b4, RT), render_only), b4);
    expect("b5g5r5a1 RT, no stand-in", packed_target_stand_in(texture(b5, RT), bare), b5);
    expect("rgba8 RT", packed_target_stand_in(texture(rgba, RT), none), rgba);
    expect("b5g5r5a1 sampled only", packed_target_stand_in(texture(b5, 0), none), b5);
    expect("b5g5r5a1 UAV without RT", packed_target_stand_in(texture(b5, UA), none), b5);
    expect("b4g4r4a4 3D RT", packed_target_stand_in(texture(b4, RT, D3D12_RESOURCE_DIMENSION_TEXTURE3D), none), b4);
    expect("b4g4r4a4 1D RT", packed_target_stand_in(texture(b4, RT, D3D12_RESOURCE_DIMENSION_TEXTURE1D), none), b4);
    expect("buffer", packed_target_stand_in(texture(b5, UA, D3D12_RESOURCE_DIMENSION_BUFFER), none), b5);
    D3D12_RESOURCE_DESC shaped = texture(b5, RT);
    shaped.SampleDesc.Count = 4;
    expect("b5g5r5a1 RT multisampled", packed_target_stand_in(shaped, none), b5);
    shaped = texture(b565, RT);
    shaped.SampleDesc.Count = 2;
    expect("b5g6r5 RT multisampled", packed_target_stand_in(shaped, none), b565);

    // B5G6R5_UNORM follows the same rule as the other two packed formats.
    expect("b5g6r5 RT, unsupported", packed_target_stand_in(texture(b565, RT), none), rgba);
    expect("b5g6r5 RT|UAV, render only", packed_target_stand_in(texture(b565, RT | UA), render_only), rgba);
    expect("b5g6r5 RT, supported", packed_target_stand_in(texture(b565, RT), full), b565);
    expect("b5g6r5 RT|UAV, supported", packed_target_stand_in(texture(b565, RT | UA), full), b565);
    expect("b5g6r5 RT, no stand-in", packed_target_stand_in(texture(b565, RT), bare), b565);
    expect("b5g6r5 sampled only", packed_target_stand_in(texture(b565, 0), none), b565);
    expect("b5g6r5 3D RT", packed_target_stand_in(texture(b565, RT, D3D12_RESOURCE_DIMENSION_TEXTURE3D), none), b565);
    expect("b5g6r5 depth-stencil", packed_target_stand_in(texture(b565, RT | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), none), b565);

    // Arrays and cube maps (six slices): the format follows the same rule for 2 to 2048 slices; 0 and
    // more than the limit keep the format. The slice count is not part of the answer.
    for (const UINT slices : {2u, 6u, 16u, 2048u}) {
        shaped = texture(b5, RT);
        shaped.DepthOrArraySize = static_cast<UINT16>(slices);
        char name[64];
        std::snprintf(name, sizeof(name), "b5g5r5a1 RT array of %u, unsupported", slices);
        expect(name, packed_target_stand_in(shaped, none), rgba);
        std::snprintf(name, sizeof(name), "b5g5r5a1 RT array of %u, supported", slices);
        expect(name, packed_target_stand_in(shaped, full), b5);
        shaped.Format = b565;
        std::snprintf(name, sizeof(name), "b5g6r5 RT array of %u, unsupported", slices);
        expect(name, packed_target_stand_in(shaped, none), rgba);
        shaped.Format = b4;
        std::snprintf(name, sizeof(name), "b4g4r4a4 RT array of %u, render only", slices);
        expect(name, packed_target_stand_in(shaped, render_only), b4);
    }
    for (const UINT slices : {0u, 2049u, 65535u}) {
        shaped = texture(b565, RT);
        shaped.DepthOrArraySize = static_cast<UINT16>(slices);
        char name[64];
        std::snprintf(name, sizeof(name), "b5g6r5 RT with %u slices", slices);
        expect(name, packed_target_stand_in(shaped, none), b565);
    }
    shaped = texture(b5, RT | UA);
    shaped.DepthOrArraySize = 6;
    shaped.MipLevels = 3;
    expect("b5g5r5a1 RT|UAV cube with mips", packed_target_stand_in(shaped, none), rgba);
    shaped.SampleDesc.Count = 4;
    expect("b5g5r5a1 RT|UAV cube multisampled", packed_target_stand_in(shaped, none), b5);
    shaped = texture(b5, RT, D3D12_RESOURCE_DIMENSION_TEXTURE3D);
    shaped.DepthOrArraySize = 6;
    expect("b5g5r5a1 3D with depth 6", packed_target_stand_in(shaped, none), b5);
    shaped = texture(rgba, RT);
    shaped.DepthOrArraySize = 6;
    expect("rgba8 RT array", packed_target_stand_in(shaped, none), rgba);

    constexpr auto sharedexp = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    constexpr auto hdr = DXGI_FORMAT_R11G11B10_FLOAT;
    constexpr UINT sample = D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE;
    const Adapter sharedexp_missing{0, 0, rt | uav, 0, sample, rt | uav};
    const Adapter sharedexp_rt_only{0, 0, rt | uav, 0, sample | rt, rt | uav};
    const Adapter sharedexp_supported{0, 0, rt | uav, 0, sample | rt | uav, rt | uav};
    const Adapter hdr_no_rt{0, 0, rt | uav, 0, sample, uav};
    const Adapter hdr_no_uav{0, 0, rt | uav, 0, sample, rt};
    expect("shared exponent RT only", packed_target_stand_in(texture(sharedexp, RT), hdr_no_uav), hdr);
    expect("shared exponent UAV only", packed_target_stand_in(texture(sharedexp, UA), hdr_no_rt), hdr);
    expect("shared exponent RT|UAV", packed_target_stand_in(texture(sharedexp, RT | UA), sharedexp_missing), hdr);
    expect("shared exponent RT supported", packed_target_stand_in(texture(sharedexp, RT), sharedexp_rt_only), sharedexp);
    expect("shared exponent RT supported, UAV missing",
        packed_target_stand_in(texture(sharedexp, RT | UA), sharedexp_rt_only), hdr);
    expect("shared exponent RT|UAV supported",
        packed_target_stand_in(texture(sharedexp, RT | UA), sharedexp_supported), sharedexp);
    expect("shared exponent RT|UAV replacement missing RT", packed_target_stand_in(texture(sharedexp, RT | UA), hdr_no_rt), sharedexp);
    expect("shared exponent RT|UAV replacement missing UAV", packed_target_stand_in(texture(sharedexp, RT | UA), hdr_no_uav), sharedexp);
    expect("shared exponent sampled only", packed_target_stand_in(texture(sharedexp, 0), sharedexp_missing), sharedexp);
    expect("shared exponent depth stencil",
        packed_target_stand_in(texture(sharedexp, RT | UA | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), sharedexp_missing), sharedexp);
    expect("shared exponent 3D", packed_target_stand_in(texture(sharedexp, RT | UA, D3D12_RESOURCE_DIMENSION_TEXTURE3D),
        sharedexp_missing), sharedexp);
    shaped = texture(sharedexp, RT | UA);
    shaped.DepthOrArraySize = 6;
    expect("shared exponent RT|UAV array", packed_target_stand_in(shaped, sharedexp_missing), hdr);
    shaped.SampleDesc.Count = 4;
    expect("shared exponent multisampled", packed_target_stand_in(shaped, sharedexp_missing), sharedexp);

    // Views: a packed view becomes RGBA8 only on the stand-in for that very format.
    using x2w_startup::PackedView;
    constexpr DXGI_FORMAT plain = DXGI_FORMAT_UNKNOWN;
    view("b5g5r5a1 view of its stand-in", packed_view(b5, rgba, b5), PackedView::stand_in);
    view("b4g4r4a4 view of its stand-in", packed_view(b4, rgba, b4), PackedView::stand_in);
    view("b5g5r5a1 view of a b5g5r5a1 texture", packed_view(b5, b5, plain), PackedView::unchanged);
    view("srgb view of an rgba8 texture", packed_view(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, rgba, plain), PackedView::unchanged);
    view("rgba8 view of a stand-in", packed_view(rgba, rgba, b5), PackedView::unchanged);
    view("b5g5r5a1 view of a plain rgba8 texture", packed_view(b5, rgba, plain), PackedView::refused);
    view("b5g5r5a1 view of the b4g4r4a4 stand-in", packed_view(b5, rgba, b4), PackedView::refused);
    view("b5g5r5a1 view of a b4g4r4a4 texture", packed_view(b5, b4, plain), PackedView::refused);
    view("b5g5r5a1 view of a bgra8 texture", packed_view(b5, DXGI_FORMAT_B8G8R8A8_UNORM, plain), PackedView::refused);
    view("b5g6r5 view of its stand-in", packed_view(b565, rgba, b565), PackedView::stand_in);
    view("b5g6r5 view of a b5g6r5 texture", packed_view(b565, b565, plain), PackedView::unchanged);
    view("b5g6r5 view of a plain rgba8 texture", packed_view(b565, rgba, plain), PackedView::refused);
    view("b5g6r5 view of the b5g5r5a1 stand-in", packed_view(b565, rgba, b5), PackedView::refused);
    view("b5g5r5a1 view of the b5g6r5 stand-in", packed_view(b5, rgba, b565), PackedView::refused);
    view("b5g6r5 view of a b5g5r5a1 texture", packed_view(b565, b5, plain), PackedView::refused);

    // Provenance, as the view paths compose it: packed_view(view format, the texture's format,
    // packed_stand_in_of(texture)). A texture created as R8G8B8A8_UNORM for a packed format reads
    // back that packed format and its packed view becomes the stand-in; a texture created in the
    // packed format itself carries no tag, so its own view is unchanged; an untagged R8G8B8A8_UNORM
    // texture refuses the packed view.
    view("shared exponent view of a tagged rgba8 texture", packed_view(sharedexp, rgba, sharedexp), PackedView::refused);
    view("shared exponent view of another stand-in", packed_view(sharedexp, rgba, b5), PackedView::refused);
    for (const DXGI_FORMAT packed : {b5, b4, b565, sharedexp}) {
        const auto replacement = packed == sharedexp ? hdr : rgba;
        char name[64];
        Texture stand_in_texture(replacement);
        x2w_startup::mark_packed_stand_in(&stand_in_texture, packed);
        std::snprintf(name, sizeof(name), "format %d stand-in tag", packed);
        expect(name, packed_stand_in_of(&stand_in_texture), packed);
        std::snprintf(name, sizeof(name), "format %d view of its tagged stand-in", packed);
        view(name, packed_view(packed, stand_in_texture.GetDesc().Format, packed_stand_in_of(&stand_in_texture)), PackedView::stand_in);
        Texture described(packed);
        x2w_startup::mark_packed_stand_in(&described, packed);
        std::snprintf(name, sizeof(name), "format %d texture created as described, tag", packed);
        expect(name, packed_stand_in_of(&described), plain);
        std::snprintf(name, sizeof(name), "format %d view of a texture created as described", packed);
        view(name, packed_view(packed, described.GetDesc().Format, packed_stand_in_of(&described)), PackedView::unchanged);
        Texture untagged(replacement);
        std::snprintf(name, sizeof(name), "format %d view of an untagged replacement texture", packed);
        view(name, packed_view(packed, untagged.GetDesc().Format, packed_stand_in_of(&untagged)), PackedView::refused);
    }
    Texture b4_stand_in(rgba);
    x2w_startup::mark_packed_stand_in(&b4_stand_in, b4);
    view("b5g5r5a1 view of the tagged b4g4r4a4 stand-in", packed_view(b5, b4_stand_in.GetDesc().Format, packed_stand_in_of(&b4_stand_in)),
        PackedView::refused);

    if (failures) {
        std::printf("startup_packed_target: %d FAILED\n", failures);
        return 1;
    }
    std::printf("startup_packed_target: PASS (%d rows)\n", checks);
    return 0;
}
