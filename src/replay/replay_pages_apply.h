// Memory pages for replay (docs/startup-runtime.md#d3d12-replay-record): the hook the player calls before every call,
// filled by the page reader (PAGE/MAP chunks). It writes every page image recorded with a sequence number below `seq`
// that it has not written yet.
#pragma once

#include "replay_file.h"

namespace x2w_replay {

inline void apply_pages_before(const Recording& r, u64 seq)
{
    (void)r, (void)seq;
}

}  // namespace x2w_replay
