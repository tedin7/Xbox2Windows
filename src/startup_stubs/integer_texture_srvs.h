#pragma once

#include <bitset>
#include <cstdio>
#include <d3d12.h>
#include <string>
#include <vector>

namespace x2w_startup {

// Twin registers and their flags are appended to the ordinary SRVs and entry
// values. The original shader resource indices retain their table meaning.
// The register in srvint names the UINT view. Its SINT view follows at reg + 1.
// The entry value selects float (0), unsigned (1), or signed (2).
struct IntegerTextureSrvs {
    struct Twin {
        UINT input, reg, entry;
    };
    std::vector<Twin> twins;
    bool typed = false;
    bool filtered_logged = false;

    bool read_twin(const std::string& line)
    {
        Twin twin{};
        int end = 0;
        if (std::sscanf(line.c_str(), "srvint %u %u %u%n", &twin.input, &twin.reg, &twin.entry, &end) != 3 ||
            static_cast<std::size_t>(end) != line.size() || twins.size() >= 32)
            return false;
        typed = true;
        twins.push_back(twin);
        return true;
    }

    template <typename IsTexture>
    bool valid(UINT inputs, UINT entries, IsTexture is_texture) const
    {
        if (entries + twins.size() > 32) return false;
        std::bitset<32> seen;
        for (UINT i = 0; i < twins.size(); ++i) {
            const auto& twin = twins[i];
            if (twin.input >= inputs || twin.input >= seen.size() || !is_texture(twin.input) || seen.test(twin.input) ||
                twin.reg != inputs + i * 2 || twin.entry != entries + i)
                return false;
            seen.set(twin.input);
        }
        return true;
    }

    UINT descriptors() const { return static_cast<UINT>(twins.size()) * 2; }

    const Twin* find(UINT input) const
    {
        for (const auto& twin : twins)
            if (twin.input == input) return &twin;
        return nullptr;
    }
};

inline D3D12_SHADER_RESOURCE_VIEW_DESC null_texture_srv(D3D12_SRV_DIMENSION dimension, DXGI_FORMAT format)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = format;
    desc.ViewDimension = dimension;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // MostDetailedMip and MipLevels occupy the same words in each texture view.
    desc.Texture2D.MipLevels = 1;
    if (dimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY) desc.Texture2DArray.ArraySize = 1;
    return desc;
}

}  // namespace x2w_startup
