// Device slot 41, CreateCommandSignature: which argument descriptions desktop D3D12 cannot create.
//
// The captured console argument type 0x69 is recognised before the desktop call. Other values are
// left to the desktop runtime, whose enumeration can grow beyond DISPATCH_MESH.
//
// The header has no Windows dependency: it works on any description type that has a `Type` member.
#pragma once

#include <cstddef>

namespace x2w_startup {
template <class Description>
inline bool has_console_indirect_arg_type(const Description* arguments, std::size_t count)
{
    if (!arguments) return false;
    for (std::size_t i = 0; i < count; ++i)
        if (static_cast<unsigned>(arguments[i].Type) == 0x69) return true;
    return false;
}
constexpr bool console_indirect_dispatch_record(unsigned stride, unsigned arguments)
{
    return stride == 12 && arguments == 1;
}

constexpr bool indirect_ranges_overlap(unsigned long long first, unsigned long long bytes,
                                       unsigned long long other, unsigned long long other_bytes)
{
    return bytes && other_bytes && (first <= other ? other - first < bytes : first - other < other_bytes);
}
}  // namespace x2w_startup
