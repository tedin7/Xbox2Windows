#pragma once

#include "clear_value_format.h"

#include <windows.h>

#include <array>
#include <tuple>

namespace x2w_startup {

using FormatLog = void (*)(const char*, ...);

inline bool log_at_power_of_two(UINT64 count)
{
    return count && !(count & (count - 1));
}

inline auto resource_size(const D3D12_RESOURCE_DESC& desc)
{
    return std::make_tuple(desc.Width, desc.Height, desc.DepthOrArraySize);
}

inline void log_console_format(const char* path, int slot, const D3D12_RESOURCE_DESC& desc,
                               const D3D12_CLEAR_VALUE* clear, const D3D12_CLEAR_VALUE* mapped_clear,
                               FormatLog log)
{
    const auto mapped = desktop_format(desc.Format);
    if (mapped == desc.Format) return;

    using Key = std::tuple<DXGI_FORMAT, decltype(resource_size(desc)), D3D12_RESOURCE_FLAGS>;
    static std::array<Key, 16> seen;
    static std::size_t used = 0;
    static std::array<UINT64, 5> totals{};
    static SRWLOCK lock = SRWLOCK_INIT;

    const Key key{desc.Format, resource_size(desc), desc.Flags};
    AcquireSRWLockExclusive(&lock);
    const UINT64 count = ++totals[static_cast<UINT>(desc.Format) - 116];
    bool detail = used < seen.size();
    for (std::size_t i = 0; detail && i < used; ++i)
        detail = seen[i] != key;
    if (detail) seen[used++] = key;
    const bool summary = !detail && log_at_power_of_two(count);
    ReleaseSRWLockExclusive(&lock);

    if (detail)
        log("x2w-d3d12 console format %u -> desktop format %u path=%s slot=%d dim=%u size=%llux%ux%u "
            "mips=%u samples=%u flags=0x%x clear_present=%u clear_mapped=%u",
            static_cast<UINT>(desc.Format), static_cast<UINT>(mapped), path, slot, static_cast<UINT>(desc.Dimension),
            static_cast<unsigned long long>(desc.Width), desc.Height, static_cast<UINT>(desc.DepthOrArraySize),
            static_cast<UINT>(desc.MipLevels), desc.SampleDesc.Count, static_cast<UINT>(desc.Flags),
            static_cast<UINT>(clear != nullptr), static_cast<UINT>(clear && mapped_clear && mapped_clear->Format != clear->Format));
    if (summary)
        log("x2w-d3d12 console format %u -> desktop format %u count=%llu",
            static_cast<UINT>(desc.Format), static_cast<UINT>(mapped), static_cast<unsigned long long>(count));
}

// Private data follows the native resource lifetime, including aliases of the same COM object.
inline constexpr GUID console_format_tag = {0x6f1997f0, 0xbd8b, 0x4e8c, {0xbd, 0xaf, 0x54, 0xae, 0x6b, 0x23, 0x66, 0x17}};

inline void mark_console_format(ID3D12Resource* resource, DXGI_FORMAT original)
{
    if (desktop_format(original) != original)
        resource->SetPrivateData(console_format_tag, sizeof(original), &original);
}

inline bool has_console_format_remap(ID3D12Resource* resource)
{
    DXGI_FORMAT original{};
    UINT bytes = sizeof(original);
    return SUCCEEDED(resource->GetPrivateData(console_format_tag, &bytes, &original)) &&
           bytes == sizeof(original) && desktop_format(original) != original;
}

} // namespace x2w_startup
