// What a title's ClearDepthStencilView (command-list slot 51, desktop 47 + 4) may forward to the desktop list.
// Plain C++ with no Windows types, so the decision is checked without a device (tests/list_clear_depth_check.cpp).
//
// The desktop call is only valid for a plane the view can write, a depth value in [0, 1] when depth is cleared, and
// a rectangle array that can be read. Anything else would make the debug layer report an error or remove the
// device, so it is refused here and counted by the caller.
#pragma once

#include <cmath>

namespace x2w_clear_depth {

// D3D12_CLEAR_FLAG_DEPTH and D3D12_CLEAR_FLAG_STENCIL; the caller checks the values against the SDK.
constexpr unsigned clear_depth = 1;
constexpr unsigned clear_stencil = 2;

enum class Verdict {
    forward,           // call the desktop list with Plan::flags and Plan::rect_count
    nothing_to_clear,  // no plane left to clear after normalising: not an error, not forwarded
    bad_depth,         // depth is cleared and the value is not a number in [0, 1]
    read_only_plane,   // a plane to clear is read-only in this view
    unreadable_rects,  // rects were announced but cannot be read
};

struct Request {
    unsigned flags = 0;
    float depth = 0.0f;
    bool format_has_stencil = false;
    bool depth_read_only = false;
    bool stencil_read_only = false;
    unsigned rect_count = 0;
    bool rects_readable = true;  // meaningful only when rect_count is not zero
};

struct Plan {
    Verdict verdict = Verdict::nothing_to_clear;
    unsigned flags = 0;
    unsigned rect_count = 0;
};

inline Plan plan(const Request& request)
{
    Plan out;
    unsigned flags = request.flags & (clear_depth | clear_stencil);
    if (!request.format_has_stencil) flags &= ~clear_stencil;
    if (!flags) return out;
    if ((flags & clear_depth) && !(request.depth >= 0.0f && request.depth <= 1.0f)) {  // false for NaN
        out.verdict = Verdict::bad_depth;
        return out;
    }
    if (((flags & clear_depth) && request.depth_read_only) || ((flags & clear_stencil) && request.stencil_read_only)) {
        out.verdict = Verdict::read_only_plane;
        return out;
    }
    if (request.rect_count && !request.rects_readable) {
        out.verdict = Verdict::unreadable_rects;
        return out;
    }
    out.verdict = Verdict::forward;
    out.flags = flags;
    out.rect_count = request.rect_count;
    return out;
}

inline const char* reason(Verdict verdict)
{
    switch (verdict) {
    case Verdict::forward: return "forwarded";
    case Verdict::nothing_to_clear: return "no plane to clear";
    case Verdict::bad_depth: return "depth is not a number in [0, 1]";
    case Verdict::read_only_plane: return "a plane to clear is read-only in the view";
    case Verdict::unreadable_rects: return "the rectangles cannot be read";
    }
    return "unknown";
}

}  // namespace x2w_clear_depth
