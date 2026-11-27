#include "startup_stubs/viewport_check.h"
#include <cstdio>
#include <cstring>
#include <limits>
#include <initializer_list>

int main()
{
    using namespace x2w_startup;
    const float nan = (std::numeric_limits<float>::quiet_NaN)();
    const float inf = (std::numeric_limits<float>::infinity)();
    D3D12_VIEWPORT valid{ -0.0f, -10.0f, 32.0f, 20.0f, 1.0f, 0.0f };
    auto copy = valid;
    if (sanitize_viewport(copy) || std::memcmp(&valid, &copy, sizeof(copy))) return 1;
    D3D12_RECT rect{-4, -3, 8, 9}; auto saved = rect;
    if (sanitize_scissor(rect) || std::memcmp(&rect, &saved, sizeof(rect))) return 2;
    D3D12_VIEWPORT v{nan, inf, -4, nan, nan, nan};
    if (!sanitize_viewport(v) || v.TopLeftX != 0 || v.TopLeftY != 32767 || v.Width != 0 || v.Height != 0 || v.MinDepth != 0 || v.MaxDepth != 1) return 3;
    v = {-40000, 40000, inf, 8, -0.5f, 1.5f};
    if (!sanitize_viewport(v) || v.TopLeftX != -32768 || v.TopLeftY != 32767 || v.Width != 65535 || v.Height != 0 || v.MinDepth != 0 || v.MaxDepth != 1) return 4;
    v = {32766, -32767, 2, 65535, 0, 1};
    if (!sanitize_viewport(v) || v.Width != 1 || v.Height != 65534) return 5;
    rect = {10, 20, 2, 3};
    if (!sanitize_scissor(rect) || rect.right != 10 || rect.bottom != 20) return 6;
    for (bool bound : {false, true}) for (DXGI_FORMAT stream : {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_D24_UNORM_S8_UINT})
        if (draw_dsv_format(bound, DXGI_FORMAT_D32_FLOAT, stream) != (bound ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_UNKNOWN)) return 7;
    for (bool has_state : {false, true}) for (DXGI_FORMAT format : {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_D32_FLOAT})
        if (draw_depth_state_enabled(has_state, format) != (has_state && format != DXGI_FORMAT_UNKNOWN)) return 8;
    v = {0, 0, nan, -1, 2, -1};
    if (!sanitize_viewport(v) || v.Width != 0 || v.Height != 0 || v.MinDepth != 1 || v.MaxDepth != 0) return 9;
    std::puts("startup_viewport: PASS");
    return 0;
}
