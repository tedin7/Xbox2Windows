#pragma once

namespace x2w {
// A private shader-visible heap mirrors each proven guest interval once per draw.
struct PixelHeapSpan { unsigned first, last, dimension, host_first; };
inline constexpr unsigned first_pixel_heap_slot = 1024; // preceding slots reserved for the MS uint selector
}
