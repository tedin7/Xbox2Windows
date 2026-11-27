#pragma once
#include <d3d12.h>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace x2w_startup {
struct NativeTextureWindow {
    UINT mip = 0, mips = 1, slice = 0, slices = 1, plane = 0;
};

// Build from the resolved descriptors for one dispatch. A single-access
// resource needs one whole-resource state irrespective of its mip/array size.
class NativeDispatchStates {
    struct Interval { UINT first, end; bool write; };
    struct Use {
        bool read = false, write = false, upload = false;
        UINT mips = 0, slices = 0;
        std::vector<Interval> intervals;
    };
    std::map<ID3D12Resource *, Use> resources;
    std::vector<D3D12_RESOURCE_BARRIER> transitions, uavs;

  public:
    template<class Views> explicit NativeDispatchStates(const Views &views) {
        bool mixed_access = false;
        for (const auto &v : views) {
            if (!v.resource) continue;
            auto &use = resources[v.resource.Get()];
            if (v.write) use.write = true;
            else use.read = true;
            use.upload |= v.buffer.upload;
            mixed_access |= use.read && use.write;
        }
        if (mixed_access) for (const auto &v : views) {
            if (!v.resource) continue;
            auto &use = resources.at(v.resource.Get());
            if (!use.read || !use.write) continue;
            if (!use.mips) {
                const auto desc = v.resource->GetDesc();
                use.mips = desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? 1 : desc.MipLevels;
                use.slices = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1 : desc.DepthOrArraySize;
            }
            const auto &w = v.texture;
            if (!w.mip && w.mips == use.mips) {
                const UINT first = (w.slice + w.plane * use.slices) * use.mips;
                use.intervals.push_back({first, first + w.slices * use.mips, v.write});
                continue;
            }
            for (UINT a = 0; a < w.slices; ++a) {
                const UINT first = w.mip + (w.slice + a + w.plane * use.slices) * use.mips;
                use.intervals.push_back({first, first + w.mips, v.write});
            }
        }
        for (auto &[resource, use] : resources) {
            if (!use.write) continue;
            if (use.upload) throw std::runtime_error("native UAV aliases an upload heap");
            D3D12_RESOURCE_BARRIER uav{};
            uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav.UAV.pResource = resource;
            uavs.push_back(uav);
            auto &ranges = use.intervals;
            std::sort(ranges.begin(), ranges.end(), [](auto &a, auto &b) { return a.first < b.first; });
            for (size_t j = 0; j < ranges.size();) {
                auto range = ranges[j++];
                while (j < ranges.size() && ranges[j].first <= range.end) {
                    if (ranges[j].write != range.write) {
                        if (ranges[j].first < range.end)
                            throw std::runtime_error("overlapping compute SRV and UAV subresource " + std::to_string(ranges[j].first));
                        break;
                    }
                    range.end = (std::max)(range.end, ranges[j++].end);
                }
                D3D12_RESOURCE_BARRIER b{};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                b.Transition = {resource, range.first, D3D12_RESOURCE_STATE_COMMON,
                                range.write ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
                for (; b.Transition.Subresource < range.end; ++b.Transition.Subresource)
                    transitions.push_back(b);
            }
        }
    }
    bool mixed(ID3D12Resource *resource) const {
        auto i = resources.find(resource);
        return i != resources.end() && i->second.read && i->second.write;
    }
    template<class Apply> void whole_states(Apply apply) const {
        for (const auto &[resource, use] : resources)
            if (!use.upload)
                apply(resource, use.write ? (use.read ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                                          : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    const auto &split() const { return transitions; }
    const auto &writes() const { return uavs; }
    void reverse() {
        for (auto &b : transitions) std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    }
};
}
