// Experimental (docs/startup-runtime.md, #870): the stream-output bindings of a recording command list,
// set by SOSetTargets (list slot 49, desktop 45 + 4). Plain standard C++ with no Windows types, so the
// rules can be compiled and tested on their own.
//
// A list holds four stream-output slots. SOSetTargets(start, count, views) replaces the slots
// [start, start + count). With no view array the same slots are unbound. A call that does not fit
// in the four slots is refused whole and the bindings stay as they were.
#pragma once

#include <cstdint>

namespace x2w_startup {

inline constexpr unsigned stream_output_slot_count = 4;

struct StreamOutputView {
    std::uint64_t location = 0;               // BufferLocation
    std::uint64_t size = 0;                   // SizeInBytes
    std::uint64_t filled_size_location = 0;   // BufferFilledSizeLocation
};

struct StreamOutputTargets {
    StreamOutputView views[stream_output_slot_count] = {};

    unsigned bound_count() const noexcept
    {
        unsigned n = 0;
        for (const StreamOutputView& v : views) n += v.location != 0;
        return n;
    }
};

enum class StreamOutputResult { applied, unbound, refused };

// `views` may be null (unbind). When it is not null it must hold `count` entries.
inline StreamOutputResult set_stream_output_targets(StreamOutputTargets& targets, unsigned start, unsigned count,
                                                    const StreamOutputView* views) noexcept
{
    if (start >= stream_output_slot_count || count > stream_output_slot_count - start) return StreamOutputResult::refused;
    for (unsigned i = 0; i < count; ++i) targets.views[start + i] = views ? views[i] : StreamOutputView{};
    return views ? StreamOutputResult::applied : StreamOutputResult::unbound;
}

}  // namespace x2w_startup
