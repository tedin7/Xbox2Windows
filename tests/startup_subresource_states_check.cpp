#include "startup_stubs/subresource_states.h"

#include <cstdio>

namespace {
using namespace x2w_substates;

struct Recorder {
    UINT calls = 0;
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    void ResourceBarrier(UINT count, const D3D12_RESOURCE_BARRIER* batch)
    {
        ++calls;
        barriers.assign(batch, batch + count);
    }
};

bool same(const Range& a, const Range& b)
{
    return a.first_mip == b.first_mip && a.mips == b.mips &&
           a.first_slice == b.first_slice && a.slices == b.slices &&
           a.first_plane == b.first_plane && a.planes == b.planes;
}

bool check(bool condition, const char* field, int line)
{
    if (!condition) std::fprintf(stderr, "line %d: %s\n", line, field);
    return condition;
}

#define EXPECT(condition) if (!check((condition), #condition, __LINE__)) return false

bool view_ranges()
{
    D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 16, 8, 18, 0,
                             DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}};
    const Range full{0, 5, 0, 18, 0, 1};
    const Range first{0, 1, 0, 18, 0, 1};
    EXPECT(same(whole(desc), full));
    EXPECT(same(srv(desc, nullptr), full));
    EXPECT(same(rtv(desc, nullptr), first));

    D3D12_SHADER_RESOURCE_VIEW_DESC read{};
    read.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    EXPECT(same(srv(desc, &read), full));
    read.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    EXPECT(same(srv(desc, &read), full));
    read.ViewDimension = D3D12_SRV_DIMENSION_UNKNOWN;
    EXPECT(same(srv(desc, &read), full));
    read.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
    read.Texture1D = {1, UINT(-1), 0};
    EXPECT(same(srv(desc, &read), Range{1, 4, 0, 1, 0, 1}));
    read.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
    read.Texture1DArray = {1, 2, 3, 4, 0};
    EXPECT(same(srv(desc, &read), Range{1, 2, 3, 4, 0, 1}));
    read.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    read.Texture2D = {2, 1, 0, 0};
    EXPECT(same(srv(desc, &read), Range{2, 1, 0, 1, 0, 1}));
    read.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    read.Texture2DArray = {2, 9, 16, 4, 0, 0};
    EXPECT(same(srv(desc, &read), Range{2, 3, 16, 2, 0, 1}));
    read.Texture2DArray.ArraySize = 0;
    EXPECT(same(srv(desc, &read), full));
    read.Texture2DArray = {5, 1, 0, 1, 0, 0};
    EXPECT(same(srv(desc, &read), full));
    read.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
    EXPECT(same(srv(desc, &read), Range{0, 1, 0, 1, 0, 1}));
    read.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
    read.Texture2DMSArray = {3, 2};
    EXPECT(same(srv(desc, &read), Range{0, 1, 3, 2, 0, 1}));
    read.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    read.TextureCube = {1, 2, 0};
    EXPECT(same(srv(desc, &read), Range{1, 2, 0, 6, 0, 1}));
    read.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
    read.TextureCubeArray = {1, UINT(-1), 6, 2, 0};
    EXPECT(same(srv(desc, &read), Range{1, 4, 6, 12, 0, 1}));

    D3D12_RENDER_TARGET_VIEW_DESC target{};
    target.ViewDimension = D3D12_RTV_DIMENSION_BUFFER;
    EXPECT(same(rtv(desc, &target), full));
    target.ViewDimension = D3D12_RTV_DIMENSION_UNKNOWN;
    EXPECT(same(rtv(desc, &target), full));
    target.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1D;
    target.Texture1D = {1};
    EXPECT(same(rtv(desc, &target), Range{1, 1, 0, 1, 0, 1}));
    target.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1DARRAY;
    target.Texture1DArray = {1, 3, 2};
    EXPECT(same(rtv(desc, &target), Range{1, 1, 3, 2, 0, 1}));
    target.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    target.Texture2D = {2, 0};
    EXPECT(same(rtv(desc, &target), Range{2, 1, 0, 1, 0, 1}));
    target.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
    target.Texture2DArray = {2, 16, 3, 0};
    EXPECT(same(rtv(desc, &target), Range{2, 1, 16, 2, 0, 1}));
    target.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
    EXPECT(same(rtv(desc, &target), Range{0, 1, 0, 1, 0, 1}));
    target.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
    target.Texture2DMSArray = {3, 2};
    EXPECT(same(rtv(desc, &target), Range{0, 1, 3, 2, 0, 1}));

    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    desc.DepthOrArraySize = 32;
    read.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    read.Texture3D = {2, UINT(-1), 0};
    target.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
    target.Texture3D = {2, 3, 2};
    EXPECT(same(srv(desc, &read), Range{2, 4, 0, 1, 0, 1}));
    EXPECT(same(rtv(desc, &target), Range{2, 1, 0, 1, 0, 1}));
    EXPECT(overlaps(srv(desc, &read), rtv(desc, &target)));
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    EXPECT(same(whole(desc), Range{0, 1, 0, 1, 0, 1}));
    return true;
}

bool transitions()
{
    static constexpr auto rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
    static constexpr auto ps = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    static constexpr auto common = D3D12_RESOURCE_STATE_COMMON;
    const Range full{0, 3, 0, 4, 0, 1};
    const Range read{0, 2, 1, 2, 0, 1};
    const Range target{2, 1, 1, 2, 0, 1};
    EXPECT(!overlaps(read, target));
    EXPECT(!overlaps(read, Range{0, 2, 3, 1, 0, 1}));
    EXPECT(!overlaps(read, Range{1, 1, 2, 1, 1, 1}));
    EXPECT(overlaps(read, Range{1, 1, 2, 1, 0, 1}));

    auto* resource = reinterpret_cast<ID3D12Resource*>(UINT_PTR{1});
    auto split = make_split(resource, full, rt);
    D3D12_RESOURCE_STATES state{};
    EXPECT(uniform(split, &state) && state == rt);
    Recorder list;
    EXPECT(move(&list, split, read, ps));
    EXPECT(list.calls == 1 && list.barriers.size() == 4);
    const UINT expected[]{3, 4, 6, 7};
    for (UINT i = 0; i < 4; ++i) {
        const auto& barrier = list.barriers[i];
        EXPECT(barrier.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION && barrier.Flags == D3D12_RESOURCE_BARRIER_FLAG_NONE);
        const auto& t = barrier.Transition;
        EXPECT(t.pResource == resource && t.Subresource == expected[i] && t.StateBefore == rt && t.StateAfter == ps);
    }
    EXPECT(!uniform(split, &state));
    EXPECT(!move(&list, split, read, ps));
    EXPECT(!move(&list, split, target, rt));
    EXPECT(list.calls == 1);
    EXPECT(move_all(&list, split, common));
    EXPECT(list.calls == 2 && list.barriers.size() == 12);
    for (UINT i = 0; i < 12; ++i) {
        const auto& t = list.barriers[i].Transition;
        const bool sampled = i == 3 || i == 4 || i == 6 || i == 7;
        EXPECT(t.Subresource == i && t.StateBefore == (sampled ? ps : rt) && t.StateAfter == common);
    }
    EXPECT(uniform(split, &state) && state == common);
    EXPECT(!move_all(&list, split, common) && list.calls == 2);
    return true;
}
bool depth_ranges()
{
    D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 16, 8, 3, 2,
                             DXGI_FORMAT_R24G8_TYPELESS, {1, 0}};
    const Range full{0, 2, 0, 3, 0, 2};
    EXPECT(same(whole(desc), full));
    EXPECT(same(srv(desc, nullptr), full));
    EXPECT(same(dsv(desc, nullptr), Range{0, 1, 0, 3, 0, 2}));
    EXPECT(plane_count(DXGI_FORMAT_D24_UNORM_S8_UINT) == 2);
    EXPECT(plane_count(DXGI_FORMAT_R32G8X24_TYPELESS) == 2);
    EXPECT(plane_count(DXGI_FORMAT_D32_FLOAT_S8X24_UINT) == 2);
    EXPECT(plane_count(DXGI_FORMAT_D32_FLOAT) == 1);

    D3D12_DEPTH_STENCIL_VIEW_DESC view{};
    view.ViewDimension = D3D12_DSV_DIMENSION_UNKNOWN;
    EXPECT(same(dsv(desc, &view), full));
    view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1D;
    view.Texture1D = {1};
    EXPECT(same(dsv(desc, &view), Range{1, 1, 0, 1, 0, 2}));
    view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1DARRAY;
    view.Texture1DArray = {1, 1, 2};
    EXPECT(same(dsv(desc, &view), Range{1, 1, 1, 2, 0, 2}));
    view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    view.Texture2D = {1};
    EXPECT(same(dsv(desc, &view), Range{1, 1, 0, 1, 0, 2}));
    view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
    view.Texture2DArray = {1, 2, 9};
    EXPECT(same(dsv(desc, &view), Range{1, 1, 2, 1, 0, 2}));
    view.Texture2DArray.ArraySize = 0;
    EXPECT(same(dsv(desc, &view), full));
    view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
    EXPECT(same(dsv(desc, &view), Range{0, 1, 0, 1, 0, 2}));
    view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
    view.Texture2DMSArray = {1, 2};
    EXPECT(same(dsv(desc, &view), Range{0, 1, 1, 2, 0, 2}));

    D3D12_SHADER_RESOURCE_VIEW_DESC sampled{};
    sampled.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    sampled.Texture2DArray = {1, 1, 1, 2, 1, 0};
    sampled.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    EXPECT(same(srv(desc, &sampled), Range{1, 1, 1, 2, 0, 1}));
    sampled.Format = DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    EXPECT(same(srv(desc, &sampled), Range{1, 1, 1, 2, 1, 1}));
    sampled.Format = DXGI_FORMAT_UNKNOWN;
    EXPECT(same(srv(desc, &sampled), Range{1, 1, 1, 2, 1, 1}));
    sampled.Texture2DArray.PlaneSlice = 2;
    EXPECT(same(srv(desc, &sampled), full));
    desc.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
    sampled.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
    sampled.Texture2DMSArray = {1, 2};
    sampled.Format = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    EXPECT(same(srv(desc, &sampled), Range{0, 1, 1, 2, 0, 1}));
    sampled.Format = DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
    EXPECT(same(srv(desc, &sampled), Range{0, 1, 1, 2, 1, 1}));

    D3D12_RENDER_TARGET_VIEW_DESC color{};
    color.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    color.Texture2D = {1, 1};
    EXPECT(same(rtv(desc, &color), Range{1, 1, 0, 1, 1, 1}));
    return true;
}

bool depth_transitions()
{
    static constexpr auto common = D3D12_RESOURCE_STATE_COMMON;
    static constexpr auto depth_write = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    static constexpr auto depth_read = D3D12_RESOURCE_STATE_DEPTH_READ;
    static const auto shader_read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    EXPECT(attachment_state(depth_read, shader_read) == (depth_read | shader_read));
    EXPECT(attachment_state(depth_read, depth_write) == depth_read);
    EXPECT(attachment_state(depth_write, shader_read) == depth_write);
    EXPECT(attachment_state(D3D12_RESOURCE_STATE_RENDER_TARGET, shader_read) == D3D12_RESOURCE_STATE_RENDER_TARGET);
    const Range full{0, 2, 0, 3, 0, 2};
    const Range only_bound{1, 1, 1, 2, 0, 2};
    const auto depth = plane(only_bound, 0);
    const auto stencil = plane(only_bound, 1);
    const auto mixed = bind_depth(only_bound, D3D12_DSV_FLAG_READ_ONLY_STENCIL);
    EXPECT(writable_overlap(mixed, depth));
    EXPECT(writable_overlap(mixed, full));
    EXPECT(!writable_overlap(mixed, stencil));
    EXPECT(!writable_overlap(mixed, Range{0, 1, 1, 2, 0, 1}));
    const auto inverse = bind_depth(only_bound, D3D12_DSV_FLAG_READ_ONLY_DEPTH);
    EXPECT(!writable_overlap(inverse, depth));
    EXPECT(writable_overlap(inverse, stencil));
    EXPECT(writable_overlap(bind_depth(only_bound, D3D12_DSV_FLAG_NONE), full));
    EXPECT(!writable_overlap(bind_depth(only_bound, D3D12_DSV_FLAG_READ_ONLY_DEPTH |
                                                    D3D12_DSV_FLAG_READ_ONLY_STENCIL), full));
    Recorder list;
    auto split = make_split(nullptr, full, common);
    EXPECT(move(&list, split, depth, depth_write));
    EXPECT(list.calls == 1 && list.barriers.size() == 2);
    EXPECT(list.barriers[0].Transition.Subresource == 3 && list.barriers[1].Transition.Subresource == 5);
    EXPECT(move(&list, split, stencil, depth_read));
    EXPECT(list.calls == 2 && list.barriers.size() == 2);
    EXPECT(list.barriers[0].Transition.Subresource == 9 && list.barriers[1].Transition.Subresource == 11);
    const auto read_stencil = [&](UINT index, D3D12_RESOURCE_STATES) {
        return sampled_state(split, index, mixed, shader_read);
    };
    EXPECT(sampled_state(split, 3, inverse, shader_read) == (depth_read | shader_read));
    EXPECT(move_each(&list, split, plane(full, 1), read_stencil));
    EXPECT(list.calls == 3 && list.barriers.size() == 6);
    for (UINT i = 0; i < 6; ++i) {
        const auto& t = list.barriers[i].Transition;
        const bool bound = i == 3 || i == 5;
        EXPECT(t.Subresource == 6 + i &&
               t.StateBefore == (bound ? depth_read : common) &&
               t.StateAfter == (bound ? depth_read | shader_read : shader_read));
    }
    EXPECT(split.state[3] == depth_write && split.state[5] == depth_write);
    EXPECT(!move_each(&list, split, plane(full, 1), read_stencil));
    EXPECT(move_all(&list, split, common));
    EXPECT(list.calls == 4 && list.barriers.size() == 8);
    D3D12_RESOURCE_STATES after{};
    EXPECT(uniform(split, &after) && after == common);
    return true;
}

}  // namespace

int main()
{
    return view_ranges() && transitions() && depth_ranges() && depth_transitions() ? 0 : 1;
}
